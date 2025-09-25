# Configuration des Variables d'Environnement Render

## 🚨 VARIABLES OBLIGATOIRES 

Sur le dashboard Render, dans **Settings > Environment**, ajoutez ces variables :

### 1. Base de données
```
DATABASE_URL = [URL PostgreSQL générée automatiquement par Render]
```
⚠️ **Important** : Cette variable est automatiquement créée quand vous liez une base PostgreSQL

### 2. Clé API Groq
```
GROQ_API_KEY = gsk_your_groq_api_key_here
```

### 3. Credentials Google Cloud (base64)
```
GOOGLE_APPLICATION_CREDENTIALS_JSON = ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAidHJhbnNsYXRlLWphYXJpIiwKICAicHJpdmF0ZV9rZXlfaWQiOiAiMDY1ZmE3NjRiZThhZjczMWMwMzVjYzY5Mzk4ODQzMTE4ZjgyZGVjNSIsCiAgInByaXZhdGVfa2V5IjogIi0tLS0tQkVHSU4gUFJJVkFURSBLRVktLS0tLVxuTUlJRXZRSUJBREFOQmdrcWhraUc5dzBCQVFFRkFBU0NCS2N3Z2dTakFnRUFBb0lCQVFDOG9rTXl5b3RuRDZBeFxuQ1dkUHF1MmttRGJBL3RoZVZvaTZVaDhHVGw1cWJKVU5oQW5TMmtxZWtuTVBXMmtxOFQ4STc1TStNdnF4NHBzRFxuM0pHKzhLRmsyaVN3TlB5UHEvWk5rYUxicW15b1IwdHVWWi9rRUNMdkNOc3BhM1oyU3gwMVBiclpKWHcrQWh1clxuanRYSDBWS0xmN1pyV21tOW8rV0lBZnBHeGVLcmRLV1pGQjZKcHpJUGkvaSsrclBsem1lbkRXRUVEcitSNDdIWlxuWCtFaDR3SzhOTWhFSjlxd3dCMGtuUEtmaks2MktZbmVUdXVGSmt1OUREREtOdGIrU3piT1c3Z2x0VVAwOElORFxuUW1KVzRsSFUrbFphcTIrVExZZEZUbE4rM3FQWUc1NkdreE5LaXZXcWVkWjYzbWNBajdIZENqclIrNEdLMlRIalxuTCszYWhoZHJBZ01CQUFFQ2dnRUFHNllVLzI3Y3ZBQjRWSGV5WXVVRTdDazdiVkp0V0hhS2wvVE5nMWtkVWJBVVxuc09SM3BDWkc1Yk5GbGl5cDM1ZDR5OENCM280T1hPQ0FKRExDVWlCbko2Y3AxWUlPdXVoVHM4N3k5Z3VtM1l1Rlxuck9oMEJUT3dTeVViS3BrTHhnOEFldnRQVmVDK3MwQVAyamVYY2pCbVR2Z3JQdFVEeHI0dE9LQkJsUG9RWFZ6OVxuTGhDZjdkMnZBQnVQbW4xSUsvcFhoQ1hVUXNxTHFoRUF0V2Q4UytiUXNCVk9USmNHNFpiWEMyYk51bU9LVVhqUFxuOXFaZjdIdmhLVzVxSnZEREVOMkYrMitPb1pxQ0pIVE9YdXhMUjJaY2FRbGJiM1FScEZ0Yis3d3hScGFHdTdWbVxuMnRKa3lJalJzV3pZTXJSSWpwaWIyUm12bU9mdkhYYkNQUWt4c0I0YklRS0JnUURpVXlYUFo0T00xTmExYlhJMFxuazBYMkJLSEczSkFidW1VL283cFhhcHhXbmUwbDRreVBZNTNQSDF5TWpQRFlXV1c3bmNFaDIxZWl3MVhjQ1hkeVxuUjExdjhmTUk2aEcrcS9YTGFucVNxVGpta1RraG90cFpsVU11MFBYdzRkMXpwQ2dlK0ptTkFMNStBSXNTaUl2T1xuNmJSdStlbnZUZmQ1cG5XZklDUTVFMEx3R1FLQmdRRFZYZmx5OEJuUXJLWUczL3NGZjQ3TTBWZC9Sa2ZWMzhwOFxuRHlpcCtwQjZKdmZIMnQyWm9zdG9xZ0dISStXKzRpVUtPcURvUEJGWkN6UzNiYTZNNmFlY2crUE9ScVE0WnBiQ1xuTWJoMG1NY2N0MndGR1huQVU4N3d0QlpmY3FMckxGTmV0eis3TU1JZkJQUnMzY29QSTFuV1JQOVdSRHBsY3lFNVxuQXBndmkvSGtJd0tCZ0gyRk5kUVlpZG9DZE8zOFBEbXljRHVvaC95ZGVRTVgxbTE4SmEyenYrODkrVGRva2FONFxuMFIzOU83dnJzdXVhY1JTandtZy9tUlA1RjlaSUhjbndrSDYwaG1Dc1NKa2lEOWo0UGZDM0Q0cTRnaUlJaXViaVxuSkhlRGh0Tkl3U1FRcC91OVRwUG9GTDRYR1FwM1ZtcTBMTkRicEFXSGpUZ1h2T1k2MGp3aFZaK0pBb0dCQUw4V1xucUw5ZGRXRkVIblZ0eDF6SFB0Y0ViWDVaTFNESWlvbk04Ymc1NVliZ3Uyek1BVWNGVzNMalB2Y1BHVVY1TjM4UFxudnVmK1UvTVJiQ2NUSGc2cGZDbGFuWDB5R3dWQUs2akQ2dFRSdHhGcTNGMDgwYU9ENjRQN1pVWWFYdnFMK2lhTFxucTVJME9zYVpJMmNVdzFBSGN2L0pUM3l0SWplVmdwMlNTMFhaNmN0TkFvR0FWMmNlSTRtMzErOTZHZzlrdTdWNVxuYzBMYXBqN040bXI4Kyt1Q1AzTWRsMzBvc3o0SEtPYlI4YWZDUXIzaEswck91U2RUaUhQWnhDcXRXcDVaMEZDalxuWjZJRXJSbWxvUkNiZms5cUZ3bjFaalJWcjFxSjZhNUhwY0xiSnB6N0prZnRhS252Z2NuWU14SzJ5VWdVUE1PQVxuTmJwOTVVdTk0Y1B5VmFyUHk0eXhqU0E9XG4tLS0tLUVORCBQUklWQVRFIEtFWS0tLS0tXG4iLAogICJjbGllbnRfZW1haWwiOiAiamFhcmktNDg1QHRyYW5zbGF0ZS1qYWFyaS5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgImNsaWVudF9pZCI6ICIxMDM2NDE0NDQ0MDQ3MzI5MTcwOTUiLAogICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsCiAgInRva2VuX3VyaSI6ICJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L2phYXJpLTQ4NSU0MHRyYW5zbGF0ZS1qYWFyaS5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgInVuaXZlcnNlX2RvbWFpbiI6ICJnb29nbGVhcGlzLmNvbSIKfQo=
```

### 4. Clé secrète JWT
```
SECRET_KEY = Oy-eDEX0HB1ALOHIJkrpObRrVvI3BH8BykYPuJnF2bA
```

## 📝 VARIABLES OPTIONNELLES

### Configuration d'environnement
```
ENVIRONMENT = production
DEBUG = False
LOG_LEVEL = INFO
```

### Configuration admin par défaut
```
DEFAULT_ADMIN_EMAIL = admin@jaari.com
DEFAULT_ADMIN_USERNAME = admin
DEFAULT_ADMIN_PASSWORD = VotreMotDePasseSécurisé123!
DEFAULT_ADMIN_FULL_NAME = Administrateur Jaari
```

## 🔗 ÉTAPES DE CONFIGURATION SUR RENDER

### 1. Créer la base de données PostgreSQL
1. Sur Render.com, cliquez **"New +"** → **"PostgreSQL"**
2. Nom : `jaari-postgres`
3. Database Name : `jaari_db`
4. User : `jaari_user`
5. Cliquez **"Create Database"**

### 2. Créer le service Web
1. Cliquez **"New +"** → **"Web Service"** 
2. Connectez votre repo GitHub : `abdoulahadi/jaari-rag-api`
3. **Runtime** : Docker (ou Python 3 si vous préférez)
4. **Region** : Oregon (ou plus proche de vous)
5. **Plan** : Starter (gratuit)

### 3. Configuration Build & Deploy
```
Build Command: (automatique avec Docker)
Start Command: (automatique avec Docker) 
```

**OU si vous utilisez Python natif :**
```
Build Command: pip install -r requirements.txt
Start Command: python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 4. Configurer les variables d'environnement
Dans **Settings > Environment**, ajoutez toutes les variables listées ci-dessus.

### 5. Lier la base de données
Dans **Settings > Environment** :
- Cliquez **"Add from Database"**
- Sélectionnez votre base `jaari-postgres` 
- Cela créera automatiquement `DATABASE_URL`

### 6. Déployer
Cliquez **"Manual Deploy"** → **"Deploy latest commit"**

## 🚀 VÉRIFICATION POST-DÉPLOIEMENT

Une fois déployé, testez ces endpoints :
- `https://votre-app.onrender.com/health` → Doit retourner 200 OK
- `https://votre-app.onrender.com/docs` → Interface Swagger
- `https://votre-app.onrender.com/api/v1/chat/` → API de chat

## 🆘 RÉSOLUTION DE PROBLÈMES

### Erreur "ValueError: invalid literal for int() with base 10: 'port'"
- ✅ **Solution** : Vérifiez que `DATABASE_URL` est correctement configurée
- ✅ **Format attendu** : `postgresql://user:password@host:5432/database`

### Erreur "No open ports detected"  
- ✅ **Solution** : L'application doit écouter sur `0.0.0.0:$PORT`
- ✅ **Vérification** : Le Dockerfile utilise maintenant `CMD python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Import errors
- ✅ **Solution** : Vérifiez que toutes les dépendances sont dans `requirements.txt`
- ✅ **Test local** : `python diagnostic.py` pour tester la configuration
