import asyncio, json, sys, time
from pathlib import Path
import httpx

label, model, port = sys.argv[1:4]
with_gpu = sys.argv[4:] == ["--gpu"]
dest = Path(__file__).parent / label
base = f"http://127.0.0.1:{port}"
samples = []
for row in (
    Path("/validation/datasets/seedtts/en/meta.lst")
    .read_text()
    .splitlines()[: 16 if with_gpu else 128]
):
    sample, reftext, reference, target = row.split("|", 3)
    samples.append(
        (
            sample,
            dict(
                model=model,
                input=target,
                ref_text=reftext,
                ref_audio=Path(reference).as_uri(),
                stream=True,
                response_format="pcm",
            ),
        )
    )


async def main():
    results = []
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        profile = {
            "run_id": ("gpu-" if with_gpu else "tail-") + label,
            "event_dir": str(dest / ("gpu-events" if with_gpu else "events")),
            "event_buffer_capacity": 262144,
        }
        path = "profile" if with_gpu else "request_profile"
        first_seen = asyncio.Event()
        resp = await client.post(base + "/start_" + path, json=profile)
        resp.raise_for_status()
        sem = asyncio.Semaphore(16)

        async def generate(sample, payload):
            async with sem:
                start = time.monotonic_ns()
                epoch_start = time.time_ns()
                first = None
                size = 0
                async with client.stream(
                    "POST", base + "/v1/audio/speech", json=payload
                ) as res:
                    if res.is_error:
                        raise RuntimeError((await res.aread())[:256])
                    rid = res.headers["x-request-id"]
                    async for chunk in res.aiter_bytes():
                        if chunk and first is None:
                            first = time.monotonic_ns()
                            first_seen.set()
                        size += len(chunk)
                assert size and first
                results.append(
                    dict(
                        sample=sample,
                        request_id=rid,
                        start_ns=start,
                        start_epoch_ns=epoch_start,
                        first_ns=first,
                        end_ns=time.monotonic_ns(),
                        pcm_bytes=size,
                    )
                )

        jobs = asyncio.gather(*(generate(s, p) for s, p in samples))
        try:
            if with_gpu:
                await asyncio.wait_for(first_seen.wait(), timeout=30)
                await asyncio.sleep(0.15)
            else:
                await jobs
        finally:
            resp = await client.post(
                base + "/stop_" + path, json={"run_id": profile["run_id"]}
            )
            resp.raise_for_status()
        await jobs
        (
            dest / ("gpu-http-events.json" if with_gpu else "http-events.json")
        ).write_text(json.dumps(results))
        if with_gpu:
            import gzip

            for _ in range(90):
                paths = list((dest / "traces").rglob("*.gz"))
                if len(paths) >= 3:
                    try:
                        for p in paths:
                            with gzip.open(p, "rb") as f:
                                while f.read(1024 * 1024):
                                    pass
                        break
                    except (EOFError, OSError):
                        pass
                await asyncio.sleep(1)
        else:
            await asyncio.sleep(3)


asyncio.run(main())
