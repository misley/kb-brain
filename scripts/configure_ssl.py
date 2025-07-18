#!/usr/bin/env python3
"""
SSL Configuration Script - Secure replacement for hardcoded SSL setup
Configures SSL certificates using environment variables and secure defaults
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

# Add kb_brain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb_brain.config.settings import Settings

def configure_ssl_certificates():
    """Configure SSL certificates with secure defaults"""
    print("🔒 KB Brain SSL Configuration")
    print("=" * 40)
    
    # Check if SSL configuration is needed
    ssl_config = Settings.get_ssl_config()
    
    if ssl_config["cert_path"] and Path(ssl_config["cert_path"]).exists():
        print("✅ SSL certificates already configured")
        print(f"   Using: {ssl_config['cert_path']}")
        return
    
    print("🔍 Checking for SSL certificates...")
    
    # Common SSL certificate locations
    common_cert_paths = [
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/certs/ca-bundle.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/local/share/certs/ca-root-nss.crt",
        "/etc/ssl/cert.pem"
    ]
    
    system_cert_path = None
    for cert_path in common_cert_paths:
        if Path(cert_path).exists():
            system_cert_path = cert_path
            print(f"✅ Found system certificates: {cert_path}")
            break
    
    if not system_cert_path:
        print("⚠️  No system SSL certificates found")
        print("   This may cause HTTPS requests to fail")
        return
    
    # Check for custom certificate bundle
    custom_cert_path = Settings.SSL_CA_BUNDLE
    
    if custom_cert_path.exists():
        print(f"✅ Custom certificate bundle found: {custom_cert_path}")
        
        # Validate certificate bundle
        if validate_certificate_bundle(custom_cert_path):
            print("✅ Certificate bundle is valid")
        else:
            print("⚠️  Certificate bundle validation failed")
            return
    else:
        print("ℹ️  No custom certificate bundle configured")
        print("   Using system certificates")
    
    # Set environment variables for current session
    os.environ["SSL_CERT_PATH"] = system_cert_path
    if custom_cert_path.exists():
        os.environ["SSL_CA_BUNDLE"] = str(custom_cert_path)
    
    print("\n📋 SSL Configuration Summary:")
    print(f"   System certificates: {system_cert_path}")
    if custom_cert_path.exists():
        print(f"   Custom bundle: {custom_cert_path}")
    
    print("\n💡 To make this permanent, add to your .env file:")
    print(f"   SSL_CERT_PATH={system_cert_path}")
    if custom_cert_path.exists():
        print(f"   SSL_CA_BUNDLE={custom_cert_path}")

def validate_certificate_bundle(cert_path: Path) -> bool:
    """Validate certificate bundle format"""
    try:
        with open(cert_path, 'r') as f:
            content = f.read()
            
        # Basic validation - check for certificate markers
        if "BEGIN CERTIFICATE" in content and "END CERTIFICATE" in content:
            return True
        else:
            return False
            
    except Exception:
        return False

def install_custom_certificates():
    """Install custom certificates if provided"""
    print("\n🔧 Custom Certificate Installation")
    print("=" * 40)
    
    # Check for certificate files in common locations
    cert_locations = [
        Path.home() / "doi_cacert.pem",
        Path.home() / "ca-certificates.crt",
        Path.home() / "custom_certs.pem",
        Path.cwd() / "certificates" / "ca-bundle.crt"
    ]
    
    found_certs = []
    for cert_path in cert_locations:
        if cert_path.exists():
            found_certs.append(cert_path)
    
    if not found_certs:
        print("ℹ️  No custom certificates found")
        print("   If you have custom certificates, place them in:")
        print("   - ~/ca-certificates.crt")
        print("   - ~/custom_certs.pem")
        return
    
    print(f"📁 Found {len(found_certs)} custom certificate files:")
    for cert_path in found_certs:
        print(f"   - {cert_path}")
        
        if validate_certificate_bundle(cert_path):
            print("     ✅ Valid certificate bundle")
        else:
            print("     ⚠️  Invalid certificate format")
    
    # Install valid certificates
    target_path = Settings.SSL_CA_BUNDLE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if found_certs:
        # Use the first valid certificate bundle
        for cert_path in found_certs:
            if validate_certificate_bundle(cert_path):
                try:
                    shutil.copy2(cert_path, target_path)
                    print(f"✅ Installed custom certificates to: {target_path}")
                    break
                except Exception as e:
                    print(f"❌ Failed to install certificates: {e}")

def test_ssl_configuration():
    """Test SSL configuration with a sample HTTPS request"""
    print("\n🧪 Testing SSL Configuration")
    print("=" * 40)
    
    try:
        import requests
        
        # Test with system certificates
        print("Testing HTTPS connection...")
        
        ssl_config = Settings.get_ssl_config()
        
        # Configure requests session
        session = requests.Session()
        
        if ssl_config["ca_bundle"]:
            session.verify = ssl_config["ca_bundle"]
            print(f"   Using custom CA bundle: {ssl_config['ca_bundle']}")
        elif ssl_config["cert_path"]:
            session.verify = ssl_config["cert_path"]
            print(f"   Using system certificates: {ssl_config['cert_path']}")
        else:
            print("   Using default SSL verification")
        
        # Test connection
        response = session.get("https://httpbin.org/get", timeout=10)
        
        if response.status_code == 200:
            print("✅ SSL configuration is working correctly")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            
    except ImportError:
        print("⚠️  requests library not available - skipping SSL test")
    except Exception as e:
        print(f"❌ SSL test failed: {e}")
        print("   This may indicate SSL configuration issues")

def main():
    """Main SSL configuration function"""
    print("🚀 KB Brain SSL Configuration Tool")
    print("=" * 50)
    
    # Configure SSL certificates
    configure_ssl_certificates()
    
    # Install custom certificates if available
    install_custom_certificates()
    
    # Test SSL configuration
    test_ssl_configuration()
    
    print("\n✅ SSL configuration completed!")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env")
    print("2. Customize SSL settings in .env file")
    print("3. Test KB Brain functionality")

if __name__ == "__main__":
    main()