#!/bin/bash

# Script de maintenance pour Jaari RAG API Docker
# Usage: ./maintenance.sh [command] [options]

set -e

# Configuration
PROJECT_NAME="jaari-rag-api"
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Fonctions utilitaires
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Détection de l'environnement
detect_environment() {
    if docker-compose -f docker-compose.prod.yml ps >/dev/null 2>&1 && docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        echo "prod"
    elif docker-compose ps >/dev/null 2>&1 && docker-compose ps | grep -q "Up"; then
        echo "dev"
    else
        echo "none"
    fi
}

# Afficher l'aide
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  status      Afficher le statut des services"
    echo "  logs        Afficher les logs (--follow pour suivre)"
    echo "  backup      Créer une sauvegarde"
    echo "  restore     Restaurer depuis une sauvegarde"
    echo "  update      Mettre à jour l'application"
    echo "  restart     Redémarrer les services"
    echo "  stop        Arrêter les services"
    echo "  clean       Nettoyer les ressources inutilisées"
    echo "  admin       Gestion des administrateurs"
    echo "  health      Vérifier la santé des services"
    echo "  shell       Accéder au shell du conteneur API"
    echo ""
    echo "OPTIONS:"
    echo "  --follow    Suivre les logs en temps réel"
    echo "  --service   Spécifier un service (jaari-api, redis, postgres)"
    echo "  --env       Forcer l'environnement (dev/prod)"
    echo ""
    echo "Exemples:"
    echo "  $0 status"
    echo "  $0 logs --follow --service jaari-api"
    echo "  $0 backup"
    echo "  $0 admin --list"
}

# Status des services
show_status() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    log_info "Environnement détecté: $env"
    echo ""
    
    if [ "$env" = "none" ]; then
        log_warning "Aucun service Docker en cours d'exécution"
        return
    fi
    
    log_info "Statut des conteneurs:"
    docker-compose -f $compose_file ps
    
    echo ""
    log_info "Utilisation des ressources:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    
    echo ""
    log_info "Volumes utilisés:"
    docker volume ls | grep jaari || echo "Aucun volume jaari trouvé"
    
    echo ""
    log_info "Services accessibles:"
    echo "  🌐 API: http://localhost:8000"
    echo "  📚 Documentation: http://localhost:8000/docs"
    echo "  ❤️  Health: http://localhost:8000/health"
    
    if [ "$env" = "prod" ]; then
        echo "  🗄️  PostgreSQL: localhost:5432"
    fi
    echo "  📦 Redis: localhost:6379"
}

# Afficher les logs
show_logs() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    local follow_flag=""
    local service_filter=""
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    if [ "$FOLLOW_LOGS" = "true" ]; then
        follow_flag="-f"
    fi
    
    if [ "$SERVICE_FILTER" != "" ]; then
        service_filter="$SERVICE_FILTER"
    fi
    
    log_info "Affichage des logs ($env)..."
    docker-compose -f $compose_file logs $follow_flag --tail=100 $service_filter
}

# Créer une sauvegarde
create_backup() {
    local env=$(detect_environment)
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    log_info "Création d'une sauvegarde ($env)..."
    mkdir -p $BACKUP_DIR
    
    if [ "$env" = "prod" ]; then
        # Backup PostgreSQL
        log_info "Sauvegarde PostgreSQL..."
        docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U jaari_user jaari_rag_db > "$BACKUP_DIR/db_backup_$DATE.sql"
        
        # Backup volumes
        log_info "Sauvegarde des volumes..."
        docker run --rm -v jaari_uploads:/data -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/uploads_backup_$DATE.tar.gz -C /data .
        docker run --rm -v jaari_data:/data -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/data_backup_$DATE.tar.gz -C /data .
        
        log_success "Sauvegarde production créée:"
        echo "  - Base de données: $BACKUP_DIR/db_backup_$DATE.sql"
        echo "  - Uploads: $BACKUP_DIR/uploads_backup_$DATE.tar.gz"
        echo "  - Data: $BACKUP_DIR/data_backup_$DATE.tar.gz"
        
    elif [ "$env" = "dev" ]; then
        # Backup SQLite
        if [ -f "./data/jaari_rag.db" ]; then
            cp "./data/jaari_rag.db" "$BACKUP_DIR/jaari_rag_backup_$DATE.db"
            log_success "Sauvegarde dev créée: $BACKUP_DIR/jaari_rag_backup_$DATE.db"
        fi
        
        # Backup uploads
        if [ -d "./uploads" ]; then
            tar czf "$BACKUP_DIR/uploads_backup_$DATE.tar.gz" -C ./uploads .
            log_success "Uploads sauvegardés: $BACKUP_DIR/uploads_backup_$DATE.tar.gz"
        fi
    else
        log_error "Aucun service en cours, impossible de créer une sauvegarde"
        exit 1
    fi
}

# Restaurer une sauvegarde
restore_backup() {
    local env=$(detect_environment)
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    log_info "Sauvegardes disponibles:"
    ls -la $BACKUP_DIR/ | grep backup || echo "Aucune sauvegarde trouvée"
    
    read -p "Nom du fichier de sauvegarde à restaurer: " backup_file
    
    if [ ! -f "$BACKUP_DIR/$backup_file" ]; then
        log_error "Fichier de sauvegarde non trouvé: $BACKUP_DIR/$backup_file"
        exit 1
    fi
    
    log_warning "⚠️  Cette opération va écraser les données actuelles!"
    read -p "Êtes-vous sûr? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Restauration annulée"
        exit 0
    fi
    
    if [ "$env" = "prod" ] && [[ "$backup_file" == *".sql" ]]; then
        log_info "Restauration PostgreSQL..."
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U jaari_user -d jaari_rag_db < "$BACKUP_DIR/$backup_file"
        log_success "Base de données restaurée"
        
    elif [ "$env" = "dev" ] && [[ "$backup_file" == *".db" ]]; then
        log_info "Restauration SQLite..."
        cp "$BACKUP_DIR/$backup_file" "./data/jaari_rag.db"
        log_success "Base de données restaurée"
        
    else
        log_error "Type de sauvegarde non compatible avec l'environnement $env"
        exit 1
    fi
}

# Mettre à jour l'application
update_app() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    log_info "Mise à jour de l'application ($env)..."
    
    # Créer une sauvegarde avant la mise à jour
    log_info "Création d'une sauvegarde automatique..."
    create_backup
    
    # Récupérer les dernières modifications
    log_info "Récupération du code source..."
    git pull origin main
    
    # Arrêter les services
    log_info "Arrêt des services..."
    docker-compose -f $compose_file down
    
    # Reconstruire les images
    log_info "Reconstruction des images..."
    docker-compose -f $compose_file build --no-cache
    
    # Redémarrer les services
    log_info "Redémarrage des services..."
    docker-compose -f $compose_file up -d
    
    # Vérifier que tout fonctionne
    log_info "Vérification du déploiement..."
    sleep 10
    
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        log_success "Mise à jour réussie!"
        show_status
    else
        log_error "Erreur après mise à jour, vérifiez les logs"
        docker-compose -f $compose_file logs --tail=50
        exit 1
    fi
}

# Redémarrer les services
restart_services() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    if [ "$SERVICE_FILTER" != "" ]; then
        log_info "Redémarrage du service $SERVICE_FILTER..."
        docker-compose -f $compose_file restart $SERVICE_FILTER
    else
        log_info "Redémarrage de tous les services..."
        docker-compose -f $compose_file restart
    fi
    
    log_success "Services redémarrés"
}

# Arrêter les services
stop_services() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    log_info "Arrêt des services ($env)..."
    docker-compose -f $compose_file down
    log_success "Services arrêtés"
}

# Nettoyer les ressources
clean_resources() {
    log_warning "⚠️  Cette opération va supprimer les images et volumes inutilisés"
    read -p "Continuer? (y/N): " confirm
    
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log_info "Nettoyage annulé"
        exit 0
    fi
    
    log_info "Nettoyage des images inutilisées..."
    docker image prune -f
    
    log_info "Nettoyage des volumes inutilisés..."
    docker volume prune -f
    
    log_info "Nettoyage du système..."
    docker system prune -f
    
    log_success "Nettoyage terminé"
}

# Gestion des administrateurs
manage_admin() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    if [ "$env" = "none" ]; then
        log_error "Aucun service en cours d'exécution"
        exit 1
    fi
    
    log_info "Gestion des administrateurs:"
    echo "1. Lister les administrateurs"
    echo "2. Créer un nouvel administrateur"
    echo "3. Créer l'administrateur par défaut"
    
    read -p "Choisissez une option (1-3): " choice
    
    case $choice in
        1)
            docker-compose -f $compose_file exec jaari-api python create_admin.py --list
            ;;
        2)
            read -p "Email: " email
            read -p "Username: " username
            read -s -p "Password: " password
            echo
            read -p "Full Name: " fullname
            
            docker-compose -f $compose_file exec jaari-api python create_admin.py \
                --email "$email" --username "$username" --password "$password" --full-name "$fullname"
            ;;
        3)
            docker-compose -f $compose_file exec jaari-api python create_admin.py --default
            ;;
        *)
            log_error "Option invalide"
            exit 1
            ;;
    esac
}

# Vérifier la santé des services
check_health() {
    log_info "Vérification de la santé des services..."
    
    echo "🌐 API Health Check:"
    if curl -s http://localhost:8000/health | jq . 2>/dev/null; then
        log_success "API fonctionnelle"
    else
        log_error "API non accessible"
    fi
    
    echo ""
    echo "👤 Admin Status:"
    if curl -s http://localhost:8000/api/v1/auth/admin-status | jq . 2>/dev/null; then
        log_success "Auth service fonctionnel"
    else
        log_error "Auth service non accessible"
    fi
    
    echo ""
    echo "📦 Redis:"
    if docker-compose exec redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis fonctionnel"
    else
        log_error "Redis non accessible"
    fi
    
    local env=$(detect_environment)
    if [ "$env" = "prod" ]; then
        echo ""
        echo "🗄️  PostgreSQL:"
        if docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U jaari_user 2>/dev/null; then
            log_success "PostgreSQL fonctionnel"
        else
            log_error "PostgreSQL non accessible"
        fi
    fi
}

# Accéder au shell
access_shell() {
    local env=$(detect_environment)
    local compose_file="docker-compose.yml"
    
    if [ "$FORCE_ENV" != "" ]; then
        env="$FORCE_ENV"
    fi
    
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    if [ "$env" = "none" ]; then
        log_error "Aucun service en cours d'exécution"
        exit 1
    fi
    
    log_info "Accès au shell du conteneur API..."
    docker-compose -f $compose_file exec jaari-api bash
}

# Variables par défaut
FOLLOW_LOGS="false"
SERVICE_FILTER=""
FORCE_ENV=""

# Traitement des arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        status|logs|backup|restore|update|restart|stop|clean|admin|health|shell)
            COMMAND="$1"
            shift
            ;;
        --follow)
            FOLLOW_LOGS="true"
            shift
            ;;
        --service)
            SERVICE_FILTER="$2"
            shift 2
            ;;
        --env)
            FORCE_ENV="$2"
            shift 2
            ;;
        --list)
            # Pour compatibility avec admin --list
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Option inconnue: $1"
            show_help
            exit 1
            ;;
    esac
done

# Exécution de la commande
case $COMMAND in
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    backup)
        create_backup
        ;;
    restore)
        restore_backup
        ;;
    update)
        update_app
        ;;
    restart)
        restart_services
        ;;
    stop)
        stop_services
        ;;
    clean)
        clean_resources
        ;;
    admin)
        manage_admin
        ;;
    health)
        check_health
        ;;
    shell)
        access_shell
        ;;
    *)
        log_error "Commande requise"
        show_help
        exit 1
        ;;
esac
