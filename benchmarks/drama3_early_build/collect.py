import asyncio, json, time
from pathlib import Path
import httpx
import sys

root = Path("/data/drama3-build-ab") / sys.argv[1]
root.mkdir(exist_ok=True)
texts = [
    "The station is just across the river. We can reach it by walking along the old bridge.",
    "Please leave a message after the tone. I will return your call when the meeting ends.",
    "The weather cleared before noon, so the children took their books to the garden.",
    "Our train is arriving in five minutes. Please make sure you have your ticket ready.",
]


async def main():
    async with httpx.AsyncClient(
        base_url=sys.argv[2], timeout=60, trust_env=False
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

        async def wave(gap, rep):
            async def delayed(slot):
                await asyncio.sleep(slot * gap / 1000)
                return await req((slot + rep) % 4)

            return await asyncio.gather(*(delayed(i) for i in range(4)))

        await wave(0, 0)
        for mode in ["clean", "request_profile"]:
            params = dict(
                run_id="build-ab-" + sys.argv[1],
                event_dir=str(root / "events"),
                event_buffer_capacity=65536,
            )
            if mode == "request_profile":
                (await c.post("/start_request_profile", json=params)).raise_for_status()
            results = []
            try:
                for rep in range(12 if mode == "clean" else 4):
                    gaps = [0]
                    for gap in gaps:
                        rows = await wave(gap, rep)
                        results.append(dict(rep=rep, gap=gap, rows=rows))
                        (root / (mode + ".json")).write_text(json.dumps(results))
            finally:
                if mode == "request_profile":
                    (
                        await c.post(
                            "/stop_request_profile", json={"run_id": params["run_id"]}
                        )
                    ).raise_for_status()


asyncio.run(main())
