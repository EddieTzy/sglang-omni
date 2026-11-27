import asyncio, json, time
from pathlib import Path
import httpx
import sys

root = Path("/data/drama3-drain")
root.mkdir(exist_ok=True)
texts = [
    "The station is just across the river. We can reach it by walking along the old bridge.",
    "Please leave a message after the tone. I will return your call when the meeting ends.",
    "The weather cleared before noon, so the children took their books to the garden.",
    "Our train is arriving in five minutes. Please make sure you have your ticket ready.",
]


async def main():
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:18966", timeout=60, trust_env=False
    ) as c:

        async def req(i):
            d = dict(
                model="fishaudio/drama-3",
                input=texts[i],
                ref_audio="file:///references/F00A_eng_continuous.wav",
                ref_text=Path("/references/F00A_eng_continuous.txt")
                .read_text()
                .strip(),
                stream=True,
                response_format="pcm",
            )
            start = time.monotonic_ns()
            chunks = []
            async with c.stream("POST", "/v1/audio/speech", json=d) as r:
                r.raise_for_status()
                rid = r.headers["x-request-id"]
                headers = {
                    k: v
                    for k, v in r.headers.items()
                    if "rate" in k or "format" in k or "channel" in k
                }
                async for b in r.aiter_raw():
                    chunks.append((time.monotonic_ns(), len(b)))
            assert chunks
            return dict(
                headers=headers,
                i=i,
                rid=rid,
                start=start,
                end=time.monotonic_ns(),
                chunks=chunks,
            )

        async def wave(n, rep):
            return await asyncio.gather(*(req((slot + rep) % 4) for slot in range(n)))

        (root / "mode").write_text("A")
        await wave(4, 0)
        for mode in ["clean", "trace"]:
            pid = "drain_1127_" + mode
            if mode == "trace":
                (
                    await c.post(
                        "/start_request_profile",
                        json=dict(
                            run_id=pid,
                            event_dir=str(root / "events"),
                            event_buffer_capacity=65536,
                        ),
                    )
                ).raise_for_status()
            results = []
            try:
                for rep in range(12 if mode == "clean" else 6):
                    seq = [("A", 1), ("B", 1), ("B", 4), ("A", 4)]
                    if rep % 2:
                        seq = list(reversed(seq))
                    for arm, n in seq:
                        (root / "mode").write_text(arm + mode)
                        rows = await wave(n, rep)
                        results.append(dict(arm=arm, n=n, rep=rep, rows=rows))
                        (root / (mode + ".json")).write_text(json.dumps(results))
            finally:
                if mode == "trace":
                    (
                        await c.post("/stop_request_profile", json=dict(run_id=pid))
                    ).raise_for_status()


asyncio.run(main())
