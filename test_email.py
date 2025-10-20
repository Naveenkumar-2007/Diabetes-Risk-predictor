"""
Test script to verify SMTP email configuration
"""
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()

# Get SMTP configuration
smtp_host = os.getenv('SMTP_HOST')
smtp_port = int(os.getenv('SMTP_PORT', '587'))
smtp_username = os.getenv('SMTP_USERNAME')
smtp_password = os.getenv('SMTP_PASSWORD')
smtp_from = os.getenv('SMTP_FROM_EMAIL')
use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() in ('true', '1', 'yes')

print("=" * 60)
print("SMTP Configuration Test")
print("=" * 60)
print(f"SMTP Host: {smtp_host}")
print(f"SMTP Port: {smtp_port}")
print(f"SMTP Username: {smtp_username}")
print(f"SMTP Password: {'*' * len(smtp_password) if smtp_password else 'NOT SET'}")
print(f"SMTP From: {smtp_from}")
print(f"Use TLS: {use_tls}")
print("=" * 60)

if not smtp_host or not smtp_username or not smtp_password:
    print("❌ SMTP configuration is incomplete!")
    exit(1)

# Test email
test_recipient = smtp_username  # Send to self for testing

print(f"\n📧 Testing email to: {test_recipient}")

# Parse sender email
if '<' in smtp_from and '>' in smtp_from:
    sender_email = smtp_from.split('<')[1].split('>')[0].strip()
else:
    sender_email = smtp_from.strip() if smtp_from else smtp_username

message = MIMEText("This is a test email from your Diabetes Predictor application.\n\nIf you received this, your SMTP configuration is working correctly!")
message['Subject'] = 'Test Email - Diabetes Predictor'
message['From'] = smtp_from or smtp_username
message['To'] = test_recipient

try:
    print("\n🔄 Connecting to SMTP server...")
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.set_debuglevel(1)  # Enable detailed debug output
        
        if use_tls:
            print("🔐 Starting TLS...")
            server.starttls()
        
        print(f"🔑 Logging in as: {smtp_username}")
        server.login(smtp_username, smtp_password)
        
        print(f"📤 Sending email from {sender_email} to {test_recipient}...")
        server.sendmail(sender_email, [test_recipient], message.as_string())
    
    print("\n✅ Email sent successfully!")
    print(f"✅ Check inbox at: {test_recipient}")
    
except smtplib.SMTPAuthenticationError as e:
    print(f"\n❌ Authentication failed: {e}")
    print("   Check that your SMTP username and password are correct.")
    print("   For Gmail, you need an 'App Password', not your regular password.")
    print("   Generate one at: https://myaccount.google.com/apppasswords")
    
except smtplib.SMTPException as e:
    print(f"\n❌ SMTP error: {e}")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
