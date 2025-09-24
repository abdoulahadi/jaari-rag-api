#!/usr/bin/env python3
"""
Script utilitaire pour créer ou gérer l'utilisateur admin par défaut
Usage: python create_admin.py [email] [username] [password] [full_name]
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from app.config.database import SessionLocal
from app.services.user_service import UserService
from app.models.user import UserRole
from app.config.settings import settings


async def create_admin_user(email: str, username: str, password: str, full_name: str = None):
    """Créer un utilisateur admin"""
    user_service = UserService()
    db = SessionLocal()
    
    try:
        admin_data = {
            "email": email,
            "username": username,
            "password": password,
            "full_name": full_name or f"Admin {username}"
        }
        
        # Vérifier si l'email ou username existe déjà
        existing_email = await user_service.get_by_email(db, email)
        if existing_email:
            print(f"❌ Erreur: Un utilisateur avec l'email '{email}' existe déjà.")
            return False
            
        existing_username = await user_service.get_by_username(db, username)
        if existing_username:
            print(f"❌ Erreur: Un utilisateur avec le nom d'utilisateur '{username}' existe déjà.")
            return False
        
        # Créer l'admin
        admin_user = await user_service.create_default_admin(db, admin_data)
        
        if admin_user:
            print(f"✅ Utilisateur admin créé avec succès:")
            print(f"   📧 Email: {admin_user.email}")
            print(f"   👤 Username: {admin_user.username}")
            print(f"   🏷️  Nom complet: {admin_user.full_name}")
            print(f"   🔑 Rôle: {admin_user.role.value}")
            print(f"   🔐 API Key: {admin_user.api_key}")
            return True
        else:
            print("❌ Échec de la création de l'utilisateur admin.")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la création de l'admin: {str(e)}")
        return False
    finally:
        db.close()


async def list_admin_users():
    """Lister tous les utilisateurs admin"""
    user_service = UserService()
    db = SessionLocal()
    
    try:
        admins = await user_service.get_all(db, role=UserRole.ADMIN)
        
        if not admins:
            print("ℹ️  Aucun utilisateur admin trouvé.")
            return
        
        print(f"👥 Utilisateurs admin ({len(admins)}):")
        print("-" * 50)
        
        for admin in admins:
            status = "🟢 Actif" if admin.is_active else "🔴 Inactif"
            verified = "✅ Vérifié" if admin.is_verified else "⚠️  Non vérifié"
            last_login = admin.last_login.strftime("%Y-%m-%d %H:%M") if admin.last_login else "Jamais"
            
            print(f"ID: {admin.id}")
            print(f"📧 Email: {admin.email}")
            print(f"👤 Username: {admin.username}")
            print(f"🏷️  Nom: {admin.full_name}")
            print(f"📊 Statut: {status}")
            print(f"✓ Vérifié: {verified}")
            print(f"🕐 Dernière connexion: {last_login}")
            print(f"📅 Créé le: {admin.created_at.strftime('%Y-%m-%d %H:%M')}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des admins: {str(e)}")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Gestion des utilisateurs admin")
    parser.add_argument("--list", action="store_true", help="Lister tous les utilisateurs admin")
    parser.add_argument("--email", help="Email de l'admin à créer")
    parser.add_argument("--username", help="Nom d'utilisateur de l'admin à créer")
    parser.add_argument("--password", help="Mot de passe de l'admin à créer")
    parser.add_argument("--full-name", help="Nom complet de l'admin à créer")
    parser.add_argument("--default", action="store_true", help="Créer l'admin par défaut défini dans les settings")
    
    args = parser.parse_args()
    
    if args.list:
        print("🔍 Recherche des utilisateurs admin...")
        asyncio.run(list_admin_users())
        return
    
    if args.default:
        print("🚀 Création de l'admin par défaut...")
        email = settings.DEFAULT_ADMIN_EMAIL
        username = settings.DEFAULT_ADMIN_USERNAME
        password = settings.DEFAULT_ADMIN_PASSWORD
        full_name = settings.DEFAULT_ADMIN_FULL_NAME
    elif args.email and args.username and args.password:
        email = args.email
        username = args.username
        password = args.password
        full_name = args.full_name
    else:
        print("❌ Arguments manquants.")
        print("\nUtilisation:")
        print("  python create_admin.py --default  (utilise les paramètres du .env)")
        print("  python create_admin.py --email admin@example.com --username admin --password motdepasse [--full-name 'Nom Complet']")
        print("  python create_admin.py --list  (liste les admins existants)")
        sys.exit(1)
    
    print(f"🚀 Création de l'utilisateur admin...")
    print(f"   📧 Email: {email}")
    print(f"   👤 Username: {username}")
    print(f"   🏷️  Nom complet: {full_name}")
    
    success = asyncio.run(create_admin_user(email, username, password, full_name))
    
    if success:
        print("\n🎉 Admin créé avec succès ! Vous pouvez maintenant vous connecter au dashboard.")
    else:
        print("\n💥 Échec de la création de l'admin.")
        sys.exit(1)


if __name__ == "__main__":
    main()
