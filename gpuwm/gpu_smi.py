"""Read-only GPU inventory adapter; ROCm values retain their measured units.

The CSV projection serves existing inventory consumers. GPU-AMD-* is an
internal physical-card label from ROCm's unique ID, not a HIP UUID. Unknown
fields and unsupported telemetry fail rather than returning invented zeroes.
"""
from __future__ import annotations
import ctypes as C
import ctypes.util
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def is_amd():
    return os.environ.get('GPUWM_BACKEND', 'cuda').strip().lower() in ('amd', 'hip', 'rocm')


def is_metal():
    return os.environ.get('GPUWM_BACKEND', 'cuda').strip().lower() == 'metal'


def command():
    return [sys.executable,str(Path(__file__).resolve())] if (is_amd() or is_metal()) else ['nvidia-smi']


def executable():
    if is_metal():return sys.executable if shutil.which('ioreg') else None
    if is_amd():return sys.executable if shutil.which('rocm-smi') else None
    return shutil.which('nvidia-smi')


def _ioreg(*args):
    return subprocess.run(['ioreg',*args],check=True,capture_output=True,text=True,timeout=20).stdout


def metal_inventory():
    """Apple GPU, measured: Metal's recommended working set is the device memory;
    free = physical memory the OS reports available (vm_stat), capped at it."""
    import re,platform
    from gpuwm.metal import runtime as mrt
    st=mrt.rt().stats();total=int(st['recommended_max_working_set'])
    vm=subprocess.run(['vm_stat'],check=True,capture_output=True,text=True,timeout=20).stdout
    page=int(re.search(r'page size of (\d+) bytes',vm).group(1))
    pages=lambda k:int(re.search(rf'{k}:\s+(\d+)',vm).group(1))
    avail=(pages('Pages free')+pages('Pages inactive')+pages('Pages speculative'))*page
    free=min(total,avail)
    uuid=re.search(r'"IOPlatformUUID" = "([0-9A-F-]+)"',_ioreg('-rd1','-c','IOPlatformExpertDevice')).group(1)
    perf=_ioreg('-r','-d','1','-w','0','-c','IOAccelerator')
    m=re.search(r'"Device Utilization %"=(\d+)',perf)
    return [{'index':'0','uuid':'GPU-METAL-'+uuid.lower(),'name':mrt.rt().name,'pci.bus_id':'00000000:00:00.0',
             'memory.total':total//1048576,'memory.used':(total-free)//1048576,'memory.free':free//1048576,
             'utilization.gpu':int(m.group(1)) if m else 'N/A','driver_version':'macOS '+platform.mac_ver()[0],
             'compute_cap':'metal'}]


def metal_processes(cards):
    """Processes holding a Metal device client.  Metal contexts are unified
    graphics+compute (C+G) and macOS reports no per-process GPU memory."""
    import re
    rows=[]
    for pid,name in re.findall(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"',_ioreg('-l','-w','0','-c','AGXDeviceUserClient')):
        if pid=='0' or any(r['pid']==pid for r in rows):continue
        rows.append({'pid':pid,'gpu_uuid':cards[0]['uuid'],'gpu_bus_id':cards[0]['pci.bus_id'],'process_name':name,
                     'used_memory':'[N/A]','used_gpu_memory':'[N/A]'})
    return rows


def inventory():
    if is_metal():return metal_inventory()
    p=subprocess.run(['rocm-smi','--json','--showbus','--showproductname',
        '--showuniqueid','--showmeminfo','vram','--showuse','--showdriverversion'],
        check=True,capture_output=True,text=True,timeout=20)
    data=json.loads(p.stdout);rows=[]
    for key,value in data.items():
        if not key.startswith('card') or not key[4:].isdigit():continue
        uid=str(value['Unique ID'])
        if uid in ('N/A','0x0','0'): raise RuntimeError('ROCm unique ID unavailable')
        total=int(value['VRAM Total Memory (B)']);used=int(value['VRAM Total Used Memory (B)'])
        driver=value.get('Driver version',data.get('system',{}).get('Driver version','N/A'))
        name=value.get('Card Series',value.get('Card series',value.get('Card SKU','N/A')))
        rows.append({'index':key[4:],'uuid':'GPU-AMD-'+uid.removeprefix('0x'),
            'name':str(name),'pci.bus_id':str(value['PCI Bus']),
            'memory.total':total//1048576,'memory.used':used//1048576,
            'memory.free':max(0,total-used)//1048576,'utilization.gpu':value.get('GPU use (%)','N/A'),
            'driver_version':driver})
    if not rows:raise RuntimeError('rocm-smi listed no GPUs')
    return rows


class ProcessInfo(C.Structure):
    _fields_=[('process_id',C.c_uint32),('pasid',C.c_uint32),
              ('vram_usage',C.c_uint64),('sdma_usage',C.c_uint64),('cu_occupancy',C.c_uint32)]


def processes(cards):
    if is_metal():return metal_processes(cards)
    root=Path(os.environ.get('ROCM_PATH',os.environ.get('ROCM_HOME','/opt/rocm')))
    path=C.util.find_library('rocm_smi64') or str(root/'lib/librocm_smi64.so')
    lib=C.CDLL(path)
    def check(code):
        if code:raise RuntimeError(f'ROCm SMI process query failed with status {code}')
    lib.rsmi_init.argtypes=[C.c_uint64];check(lib.rsmi_init(0))
    try:
        count=C.c_uint32()
        check(lib.rsmi_compute_process_info_get(None,C.byref(count)))
        if not count.value:return []
        entries=(ProcessInfo*count.value)()
        check(lib.rsmi_compute_process_info_get(entries,C.byref(count)))
        rows=[]
        for entry in entries[:count.value]:
            n=C.c_uint32()
            check(lib.rsmi_compute_process_gpus_get(entry.process_id,None,C.byref(n)))
            ids=(C.c_uint32*n.value)()
            check(lib.rsmi_compute_process_gpus_get(entry.process_id,ids,C.byref(n)))
            for index in ids[:n.value]:
                local=ProcessInfo()
                check(lib.rsmi_compute_process_info_by_device_get(entry.process_id,index,C.byref(local)))
                card=next(c for c in cards if int(c['index'])==index)
                try:name=Path(f'/proc/{entry.process_id}/comm').read_text().strip()
                except OSError:name='N/A'
                rows.append({'pid':str(entry.process_id),'gpu_uuid':card['uuid'],
                    'gpu_bus_id':card['pci.bus_id'],'process_name':name,
                    'used_memory':local.vram_usage//1048576,'used_gpu_memory':local.vram_usage//1048576})
        return rows
    finally:lib.rsmi_shut_down()


def visibility_environment(identifier):
    if is_metal():return {}
    if not is_amd():return {'CUDA_VISIBLE_DEVICES':str(identifier)}
    rows=inventory()
    matches=[r for r in rows if str(identifier) in (r['uuid'],r['index'],r['pci.bus_id'])]
    if len(matches)!=1:raise ValueError('AMD visibility needs exactly one measured physical card')
    if os.environ.get('ROCR_VISIBLE_DEVICES'):
        raise ValueError('Explicit ROCr masking requires selecting a HIP ordinal directly; refusing ambiguous physical remapping')
    return {'CUDA_VISIBLE_DEVICES':matches[0]['index'],'HIP_VISIBLE_DEVICES':matches[0]['index']}


def _pmon(args):
    """nvidia-smi pmon -s um rows for Metal clients (type C+G, unmeasured fb)."""
    cards=inventory()
    for r in processes(cards):
        print(f"    0 {r['pid']:>7}   C+G     -     -     -     -     -     -     -     -   {r['process_name'][:32]}")
    sys.stdout.flush()


def _emit(args):
    if args and args[0]=='pmon':return _pmon(args)
    cards=inventory();query=next((a.split('=',1)[1] for a in args if a.startswith('--query-gpu=')),None)
    apps=next((a.split('=',1)[1] for a in args if a.startswith('--query-compute-apps=')),None)
    if not query and not apps:raise ValueError('GPU inventory adapter requires a supported --query-gpu or --query-compute-apps')
    rows=processes(cards) if apps else cards
    wanted=next((a.split('=',1)[1] for a in args if a.startswith(('--id=','-i='))),None)
    if wanted:
        selected={r['uuid'] for r in cards if wanted in (r['uuid'],r['index'],r['pci.bus_id'])}
        rows=[r for r in rows if r.get('uuid',r.get('gpu_uuid')) in selected]
    fields=(apps or query).split(',')
    writer=csv.writer(sys.stdout,lineterminator='\n')
    for row in rows:
        if any(f not in row for f in fields):raise ValueError('Unsupported ROCm inventory fields: '+','.join(f for f in fields if f not in row))
        writer.writerow([row[f] for f in fields])
    sys.stdout.flush()


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    interval=next((int(a.split('=',1)[1])/1000 for a in args if a.startswith('-lms=')),None)
    if interval is None:interval=next((int(args[i+1])/1000 for i,a in enumerate(args[:-1]) if a=='-lms'),None)
    try:
        while True:
            _emit(args)
            if interval is None:return 0
            time.sleep(interval)
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as e:
        print(('Metal' if is_metal() else 'ROCm')+' telemetry unavailable: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
