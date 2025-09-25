# 🚨 RÉSOLUTION ERREUR DATABASE_URL

## Problème
```
ValueError: invalid literal for int() with base 10: 'port'
```

Cette erreur arrive quand `DATABASE_URL` n'est pas correctement configurée sur Render.

## ✅ SOLUTION 1: Utiliser SQLite (Déploiement rapide)

**Avantages:** Simple, pas de configuration
**Inconvénients:** Pas de sauvegarde, pas de scalabilité

### Sur Render Dashboard:
**NE CONFIGUREZ PAS** la variable `DATABASE_URL` - laissez-la vide !

L'application utilisera automatiquement SQLite avec la valeur par défaut.

---

## 🎯 SOLUTION 2: Configurer PostgreSQL (Recommandé pour production)

### Étape 1: Créer la base PostgreSQL
1. Sur render.com → **"New +"** → **"PostgreSQL"**
2. **Database Name:** `jaari_db`
3. **User:** `jaari_user`  
4. **Region:** Oregon (même que votre app)
5. **Plan:** Free
6. Cliquez **"Create Database"**

### Étape 2: Récupérer l'URL de connexion
1. Allez dans votre base PostgreSQL
2. Dans **"Connections"**, copiez **"External Database URL"**
3. Format: `postgresql://jaari_user:PASSWORD@HOST:PORT/jaari_db`

### Étape 3: Configurer la variable sur votre Web Service
1. Allez dans votre Web Service → **Settings** → **Environment**
2. Ajoutez:
   ```
   DATABASE_URL = postgresql://jaari_user:PASSWORD@HOST:5432/jaari_db
   ```
   ⚠️ **Remplacez** PASSWORD, HOST par les vraies valeurs

### OU Méthode automatique:
1. **Settings** → **Environment** → **"Add from Database"**
2. Sélectionnez votre base `jaari-postgres`
3. Cela créera automatiquement `DATABASE_URL`

---

## 🚀 DÉPLOIEMENT RAPIDE (SQLite)

Si vous voulez déployer rapidement pour tester:

1. **Supprimez** la variable `DATABASE_URL` de Render (si elle existe)
2. Redéployez → L'app utilisera SQLite automatiquement
3. Testez: `https://votre-app.onrender.com/health`

---

## ⚠️ VÉRIFICATION DE LA DATABASE_URL

Format correct pour PostgreSQL:
```
postgresql://user:password@host:5432/database_name
```

Format INVALIDE (cause l'erreur):
```
postgresql://user:password@host:port/database_name  # ❌ "port" littéral
```

---

## 🔧 VARIABLES RENDER ACTUELLES REQUISES

```bash
# OBLIGATOIRES
GROQ_API_KEY = gsk_your_groq_api_key_here
SECRET_KEY = Oy-eDEX0HB1ALOHIJkrpObRrVvI3BH8BykYPuJnF2bA
GOOGLE_APPLICATION_CREDENTIALS_JSON = ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIs...

# OPTIONNEL (SQLite par défaut)
DATABASE_URL = postgresql://user:pass@host:5432/db  # Si PostgreSQL
```

Choisissez votre option et redéployez ! 🚀
