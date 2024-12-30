from flask import Flask, request, Response, jsonify
import hmac
import hashlib
import os
from typing import Dict

app = Flask(__name__)

# Load environment variables
WHATSAPP_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN', 'your_verify_token')
WHATSAPP_APP_SECRET = os.getenv('WHATSAPP_APP_SECRET', 'your_app_secret')

def verify_webhook(token: str) -> bool:
    """Verify the webhook token from WhatsApp."""
    return token == WHATSAPP_TOKEN

def verify_signature(request_data: bytes, signature_header: str) -> bool:
    """Verify that the request came from WhatsApp using the signature."""
    if not signature_header:
        return False
    
    expected_signature = hmac.new(
        WHATSAPP_APP_SECRET.encode(),
        request_data,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature_header, expected_signature)

def handle_message(message_data: Dict) -> Dict:
    """Process incoming WhatsApp message."""
    try:
        # Extract relevant information from the message
        entry = message_data['entry'][0]
        changes = entry['changes'][0]
        value = changes['value']
        message = value['messages'][0]
        
        # Extract message content
        message_body = message.get('text', {}).get('body', '')
        phone_number = message['from']
        
        # Process message (implement your logic here)
        response = {
            'message': f"Received: {message_body}",
            'phone_number': phone_number
        }
        
        return response
    except (KeyError, IndexError) as e:
        return {'error': f'Invalid message format: {str(e)}'}

@app.route('/webhook', methods=['GET'])
def verify():
    """Handle the webhook verification from WhatsApp."""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if mode and token:
        if mode == 'subscribe' and verify_webhook(token):
            return Response(challenge, status=200)
        return Response('Forbidden', status=403)
    
    return Response('Bad Request', status=400)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming messages from WhatsApp."""
    # Verify signature
    signature = request.headers.get('X-Hub-Signature-256', '').split('sha256=')[-1]
    if not verify_signature(request.get_data(), signature):
        return Response('Invalid signature', status=403)
    
    # Process the message
    data = request.get_json()
    if data.get('object') == 'whatsapp_business_account':
        response = handle_message(data)
        return Response(str(response), status=200)
    
    return Response('Invalid request', status=404)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'whatsapp-webhook'
    }), 200