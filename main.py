from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Meeting Notes Summarizer", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://your-frontend-domain.com",
        "https://*.vercel.app",
        "https://*.railway.app",
        "https://*.render.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SummaryRequest(BaseModel):
    text: str
    instructions: str

class SummaryResponse(BaseModel):
    summary: str
    success: bool
    message: str

class ShareRequest(BaseModel):
    summary: str
    recipients: List[str]
    subject: str = "Meeting Summary"

# AI Service Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Email Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

async def generate_summary_with_groq(text: str, instructions: str) -> str:
    """Generate summary using Groq API"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    prompt = f"""
    Instructions: {instructions}
    
    Meeting Transcript:
    {text}
    
    Please provide a structured summary based on the instructions above. 
    IMPORTANT: Do NOT use markdown formatting, asterisks, or bullet points. 
    Use clean, professional text with proper spacing and clear sections.
    Format the summary in a business-friendly, executive-ready style.
    """
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a professional meeting summarizer. Create clear, structured summaries in clean business text format. Do NOT use markdown, asterisks, or bullet points. Use professional language with clear sections and proper spacing."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=500, detail=f"Groq API error: {response.text}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")

async def generate_summary_with_openai(text: str, instructions: str) -> str:
    """Generate summary using OpenAI API as fallback"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    prompt = f"""
    Instructions: {instructions}
    
    Meeting Transcript:
    {text}
    
    Please provide a structured summary based on the instructions above.
    IMPORTANT: Do NOT use markdown formatting, asterisks, or bullet points. 
    Use clean, professional text with proper spacing and clear sections.
    Format the summary in a business-friendly, executive-ready style.
    """
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a professional meeting summarizer. Create clear, structured summaries in clean business text format. Do NOT use markdown, asterisks, or bullet points. Use professional language with clear sections and proper spacing."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.text}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

def send_email(recipients: List[str], subject: str, body: str) -> bool:
    """Send email using SMTP"""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
        raise HTTPException(status_code=500, detail="Email configuration incomplete")
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(SMTP_USER, recipients, text)
        server.quit()
        
        return True
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Meeting Notes Summarizer API"}

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    """Generate AI summary from text and instructions"""
    try:
        # Try Groq first, fallback to OpenAI
        try:
            summary = await generate_summary_with_groq(request.text, request.instructions)
        except:
            summary = await generate_summary_with_openai(request.text, request.instructions)
        
        return SummaryResponse(
            summary=summary,
            success=True,
            message="Summary generated successfully"
        )
        
    except Exception as e:
        return SummaryResponse(
            summary="",
            success=False,
            message=f"Error generating summary: {str(e)}"
        )

@app.post("/api/summarize-upload")
async def summarize_upload(
    file: UploadFile = File(...),
    instructions: str = Form(...)
):
    """Generate AI summary from uploaded file and instructions"""
    try:
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Generate summary
        try:
            summary = await generate_summary_with_groq(text, instructions)
        except:
            summary = await generate_summary_with_openai(text, instructions)
        
        return SummaryResponse(
            summary=summary,
            success=True,
            message="Summary generated successfully from uploaded file"
        )
        
    except Exception as e:
        return SummaryResponse(
            summary="",
            success=False,
            message=f"Error processing file: {str(e)}"
        )

@app.post("/api/share")
async def share_summary(request: ShareRequest):
    """Share summary via email"""
    try:
        # Prepare email body
        body = f"""
Meeting Summary

{request.summary}

---
Generated by AI Meeting Notes Summarizer
        """
        
        # Send email
        success = send_email(request.recipients, request.subject, body.strip())
        
        if success:
            return {"success": True, "message": "Summary shared successfully"}
        else:
            return {"success": False, "message": "Failed to send email"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "email_configured": bool(all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

