"""Measure the existing CPU PRIMA exporter in its own process."""
import argparse
from pathlib import Path

from brain_vlm.common import write


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prima-root",type=Path,required=True)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--resources",type=Path,required=True)
    a=p.parse_args()
    from brain_vlm.refinement.resources import ResourceMonitor
    from brain_vlm.prima.export import export_from_official
    monitor=ResourceMonitor();monitor.cuda=False
    try:
        with monitor:
            m=export_from_official(a.prima_root,a.model_root,a.destination)
        write(a.resources,{"resources":monitor.record,"compact_bytes":m["total_file_bytes"],"parity":m["validation"]})
    except Exception as error:
        write(a.resources,{"resources":monitor.record,"error":type(error).__name__+": "+str(error)})
        raise


if __name__=="__main__":main()
