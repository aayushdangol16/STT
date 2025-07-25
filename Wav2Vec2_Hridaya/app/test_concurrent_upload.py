import asyncio
import httpx

async def send_transcribe_request(client, model_name, audio_path):
    with open(audio_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model_name": model_name}
        response = await client.post(
            "http://localhost:8000/transcribe/", 
            data=data, 
            files=files,
            timeout=500.0  # increase timeout to 60 seconds
        )
        print(data)
        print(f"Status: {response.status_code}, Response: {response.json()}")

async def main():
    async with httpx.AsyncClient() as client:
        model_name = "base_100h_lm" 
        audio_path = "audio.wav"  
        tasks = [send_transcribe_request(client, model_name, audio_path) for _ in range(4)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
