import { NextRequest, NextResponse } from 'next/server'

// Proxy to FastAPI backend
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    
    // Forward the request to FastAPI backend
    const response = await fetch(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    })
    
    const data = await response.json()
    
    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || 'Upload failed' },
        { status: response.status }
      )
    }
    
    return NextResponse.json(data)
    
  } catch (error) {
    console.error('Upload proxy error:', error)
    return NextResponse.json(
      { error: 'Failed to connect to backend service' },
      { status: 500 }
    )
  }
}