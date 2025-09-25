#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier la configuration avant déploiement
"""
import os
import sys
import traceback
from pathlib import Path

def check_environment_variables():
    """Vérifier les variables d'environnement critiques"""
    print("🔍 Vérification des variables d'environnement...")
    
    required_vars = [
        "DATABASE_URL",
        "SECRET_KEY", 
        "GROQ_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var}: NON DÉFINIE")
        else:
            # Masquer les valeurs sensibles
            if len(value) > 20:
                display_value = f"{value[:10]}...{value[-10:]}"
            else:
                display_value = f"{value[:5]}..."
            print(f"✅ {var}: {display_value}")
    
    if missing_vars:
        print(f"\n⚠️  Variables manquantes: {', '.join(missing_vars)}")
        return False
    
    print("✅ Toutes les variables critiques sont définies")
    return True

def check_database_url():
    """Vérifier le format de DATABASE_URL"""
    print("\n🔍 Vérification de DATABASE_URL...")
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL non définie")
        return False
    
    # Tester le parsing de l'URL
    try:
        from sqlalchemy.engine.url import make_url
        url = make_url(db_url)
        print(f"✅ DATABASE_URL valide:")
        print(f"   - Dialecte: {url.drivername}")
        print(f"   - Host: {url.host}")
        print(f"   - Port: {url.port}")
        print(f"   - Database: {url.database}")
        return True
    except Exception as e:
        print(f"❌ DATABASE_URL invalide: {e}")
        return False

def check_google_credentials():
    """Vérifier les credentials Google Cloud"""
    print("\n🔍 Vérification des credentials Google Cloud...")
    
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON non définie")
        return False
    
    try:
        import base64
        import json
        
        # Décoder base64
        decoded = base64.b64decode(creds_json)
        creds = json.loads(decoded)
        
        required_fields = ["type", "project_id", "private_key", "client_email"]
        for field in required_fields:
            if field not in creds:
                print(f"❌ Champ manquant dans les credentials: {field}")
                return False
        
        print("✅ Credentials Google Cloud valides")
        print(f"   - Project ID: {creds.get('project_id')}")
        print(f"   - Client Email: {creds.get('client_email')}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des credentials: {e}")
        return False

def check_app_imports():
    """Vérifier que l'application peut être importée"""
    print("\n🔍 Vérification des imports de l'application...")
    
    try:
        # Ajouter le répertoire racine au path
        root_dir = Path(__file__).parent
        sys.path.insert(0, str(root_dir))
        
        # Tester l'import des settings
        from app.config.settings import settings
        print(f"✅ Settings importées - Environment: {settings.ENVIRONMENT}")
        
        # Tester l'import de l'app principale
        from app.main import app
        print("✅ Application FastAPI importée avec succès")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'import de l'application:")
        print(f"   {e}")
        traceback.print_exc()
        return False

def check_dependencies():
    """Vérifier les dépendances critiques"""
    print("\n🔍 Vérification des dépendances...")
    
    critical_packages = [
        "fastapi",
        "uvicorn", 
        "sqlalchemy",
        "groq",
        "google-cloud-translate"
    ]
    
    missing_packages = []
    for package in critical_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Packages manquants: {', '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Fonction principale de diagnostic"""
    print("🚀 DIAGNOSTIC JAARI RAG API")
    print("=" * 50)
    
    checks = [
        ("Variables d'environnement", check_environment_variables),
        ("Format DATABASE_URL", check_database_url),
        ("Credentials Google Cloud", check_google_credentials),
        ("Dépendances", check_dependencies),
        ("Imports application", check_app_imports)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erreur dans {name}: {e}")
            results.append((name, False))
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{len(results)} vérifications réussies")
    
    if passed == len(results):
        print("🎉 Toutes les vérifications sont passées ! L'application est prête pour le déploiement.")
        return 0
    else:
        print("⚠️  Certaines vérifications ont échoué. Corrigez les problèmes avant le déploiement.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
