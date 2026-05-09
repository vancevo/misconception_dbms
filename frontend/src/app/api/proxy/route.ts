import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // Gọi đến Kaggle Backend từ Next.js Server (Bypass CORS của trình duyệt)
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
    
    const response = await fetch(`${apiUrl}/api/dataset/sync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Kaggle API lỗi: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ status: 'error', message: error.message }, { status: 500 });
  }
}
