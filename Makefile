# Makefile pour Jaari RAG API
# Usage: make <command>

.PHONY: help install start sdocker-build: ## Construire l'image Docker
	docker build -t jaari-rag-api .

docker-run: ## Exécuter avec Docker Compose (dev)
	docker-compose up -d

docker-run-prod: ## Exécuter avec Docker Compose (production)
	docker-compose -f docker-compose.prod.yml up -d

docker-stop: ## Arrêter Docker Compose
	docker-compose down

docker-stop-prod: ## Arrêter Docker Compose (production)
	docker-compose -f docker-compose.prod.yml down

docker-logs: ## Logs Docker
	docker-compose logs -f app

docker-restart: ## Redémarrer Docker Compose
	docker-compose restart

docker-rebuild: ## Reconstruire et redémarrer
	docker-compose down && docker-compose build --no-cache && docker-compose up -d

# Scripts de déploiement automatisé
deploy-dev: ## Déploiement développement automatisé
	./deploy.sh dev

deploy-prod: ## Déploiement production automatisé
	./deploy.sh prod --backup

deploy-dev-rebuild: ## Déploiement dev avec reconstruction
	./deploy.sh dev --rebuild

deploy-prod-rebuild: ## Déploiement prod avec reconstruction et backup
	./deploy.sh prod --rebuild --backup

# Maintenance
maintenance-status: ## Afficher le statut des services
	./maintenance.sh status

maintenance-logs: ## Afficher les logs
	./maintenance.sh logs --follow

maintenance-backup: ## Créer une sauvegarde
	./maintenance.sh backup

maintenance-health: ## Vérifier la santé des services
	./maintenance.sh health

maintenance-admin: ## Gestion des administrateurs
	./maintenance.sh admin

maintenance-update: ## Mettre à jour l'application
	./maintenance.sh update logs test test-admin create-admin list-admins clean

# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help: ## Afficher cette aide
	@echo "Commandes disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer les dépendances
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt
	@echo "✅ Dépendances installées"

install-dev: install ## Installer les dépendances de développement
	$(PIP) install pytest requests httpx
	@echo "✅ Dépendances de développement installées"

start: ## Démarrer l'API en mode développement
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

start-prod: ## Démarrer l'API en mode production
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

stop: ## Arrêter l'API (si lancée en arrière-plan)
	pkill -f "uvicorn app.main:app" || true

restart: stop start ## Redémarrer l'API

logs: ## Afficher les logs de l'API
	tail -f logs/*.log

test: ## Exécuter les tests
	$(PYTHON) -m pytest tests/ -v

test-admin: ## Tester l'initialisation admin
	$(PYTHON) test_admin_init.py

create-admin: ## Créer l'admin par défaut
	$(PYTHON) create_admin.py --default

create-admin-custom: ## Créer un admin personnalisé
	@read -p "Email: " email; \
	read -p "Username: " username; \
	read -p "Password: " password; \
	read -p "Full Name: " fullname; \
	$(PYTHON) create_admin.py --email $$email --username $$username --password $$password --full-name "$$fullname"

list-admins: ## Lister tous les admins
	$(PYTHON) create_admin.py --list

db-init: ## Initialiser la base de données
	$(PYTHON) -c "from app.config.database import db_manager; db_manager.create_tables(); print('✅ Tables créées')"

db-reset: ## Réinitialiser la base de données (⚠️ DESTRUCTIF)
	@read -p "Êtes-vous sûr de vouloir supprimer toutes les données? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(PYTHON) -c "from app.config.database import db_manager; db_manager.drop_tables(); db_manager.create_tables(); print('✅ Base de données réinitialisée')"; \
	else \
		echo "❌ Opération annulée"; \
	fi

clean: ## Nettoyer les fichiers temporaires
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	@echo "✅ Fichiers temporaires supprimés"

health: ## Vérifier l'état de l'API
	@curl -s http://localhost:8000/health || echo "❌ API non accessible"

env-example: ## Copier .env.example vers .env
	cp .env.example .env
	@echo "✅ Fichier .env créé. Modifiez les valeurs selon vos besoins."

setup: env-example install db-init create-admin ## Configuration complète pour nouveau projet
	@echo "🎉 Configuration terminée! Utilisez 'make start' pour démarrer l'API."

docker-build: ## Construire l'image Docker
	docker build -t jaari-rag-api .

docker-run: ## Exécuter avec Docker
	docker-compose up -d

docker-stop: ## Arrêter Docker
	docker-compose down

docker-logs: ## Logs Docker
	docker-compose logs -f app

# Commandes utiles
check-env: ## Vérifier les variables d'environnement
	@echo "🔍 Vérification de l'environnement:"
	@echo "DATABASE_URL: ${DATABASE_URL}"
	@echo "REDIS_URL: ${REDIS_URL}"
	@echo "DEFAULT_ADMIN_EMAIL: ${DEFAULT_ADMIN_EMAIL}"

show-config: ## Afficher la configuration actuelle
	$(PYTHON) -c "from app.config.settings import settings; print(f'Database: {settings.DATABASE_URL}'); print(f'Admin Email: {settings.DEFAULT_ADMIN_EMAIL}')"
