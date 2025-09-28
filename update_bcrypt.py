#!/usr/bin/env python3
"""
Script pour tester et mettre à jour bcrypt
"""
import subprocess
import sys

def update_bcrypt():
    """Mettre à jour bcrypt et passlib"""
    try:
        print("Mise à jour de bcrypt et passlib...")
        
        # Installer les nouvelles versions
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "passlib[bcrypt]>=1.7.4", 
            "bcrypt>=4.0.0,<5.0.0",
            "--upgrade"
        ])
        
        print("✅ Mise à jour réussie!")
        
        # Test de bcrypt
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(
            schemes=["bcrypt"], 
            deprecated="auto",
            bcrypt__rounds=12
        )
        
        # Test avec un mot de passe court
        test_password = "Baye@221"
        print(f"Test avec mot de passe: {test_password} ({len(test_password)} caractères, {len(test_password.encode('utf-8'))} bytes)")
        
        hashed = pwd_context.hash(test_password)
        print(f"✅ Hachage réussi: {hashed[:50]}...")
        
        verified = pwd_context.verify(test_password, hashed)
        print(f"✅ Vérification réussie: {verified}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = update_bcrypt()
    sys.exit(0 if success else 1)