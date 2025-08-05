from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="TSA Item Checker", description="Check if items are allowed in TSA check-in or carry-on")

class ItemRequest(BaseModel):
    item: str

class TSAResponse(BaseModel):
    item: str
    check_in_allowed: bool
    carry_on_allowed: bool
    description: str

@app.get("/")
async def root():
    return {"message": "TSA Item Checker API"}

@app.post("/check-item", response_model=TSAResponse)
async def check_item(request: ItemRequest):
    try:
        # OpenRouter API configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
        
        # Prepare the prompt for TSA item checking
        prompt = f"""
        You are a TSA (Transportation Security Administration) expert. For the item "{request.item}", provide:
        1. Whether it's allowed in checked baggage (true/false)
        2. Whether it's allowed in carry-on baggage (true/false)
        3. A brief description of TSA rules for this item

        Respond ONLY in this exact JSON format:
        {{
            "check_in_allowed": true/false,
            "carry_on_allowed": true/false,
            "description": "Brief explanation of TSA rules for this item"
        }}
        """
        
        # Call OpenRouter API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openrouter/horizon-beta",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to get response from OpenRouter")
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            import json
            try:
                parsed_response = json.loads(ai_response)
                return TSAResponse(
                    item=request.item,
                    check_in_allowed=parsed_response["check_in_allowed"],
                    carry_on_allowed=parsed_response["carry_on_allowed"],
                    description=parsed_response["description"]
                )
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Failed to parse AI response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)