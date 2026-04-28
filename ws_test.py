import uvicorn
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket('/test')
async def ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('hello')
    await websocket.close()

uvicorn.run(app, host='127.0.0.1', port=8083, log_level='error')
