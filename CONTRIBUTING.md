# 🚀 Flujo de Trabajo Colaborativo con Git

## 🧠 Premisas
- La rama principal es `main`.
- Cada desarrollador trabaja en su propia rama (por tarea, bug o feature).
- Todas las contribuciones se hacen a través de **Pull Requests (PRs)**.
- El historial debe mantenerse **limpio** y entendible.

---

## ✅ Flujo General (Algoritmo Maestro)

### 1. Crear una rama para una tarea
```bash
git checkout main
git pull origin main
git checkout -b feat/nombre-tarea
```

> 🧠 Usa nombres de rama descriptivos, por ejemplo:
> - `feat/login-form`
> - `fix/crash-on-save`
> - `refactor/user-service`

---

### 2. Trabajar en la rama
Haz commits pequeños y claros. Antes de trabajar cada día, **actualiza tu rama** para evitar conflictos más adelante:

```bash
git fetch origin
git rebase origin/main
```

> ✅ Esto reubica tus commits encima de los más recientes de `main`, manteniendo el historial limpio.

Alternativa:
```bash
git pull --rebase origin main
```

---

### 3. Verifica antes de subir
- Corre pruebas, linters, etc.
- Si todo está bien:

```bash
git push origin feat/nombre-tarea
```

---

### 4. Crear Pull Request (PR)
Desde tu rama → `main`. Asegúrate de:

- Describir lo que hiciste
- Etiquetar compañeros para revisión
- Corregir lo que te sugieran

---

### 5. Después del merge
Una vez el PR fue aprobado y **mergeado**:

```bash
git checkout main
git pull origin main

git branch -d feat/nombre-tarea              # borra la rama local
git push origin --delete feat/nombre-tarea   # borra la rama remota (opcional)
```

---

## 🔁 ¿Merge o Rebase?

| Acción                     | ¿Usar Rebase? | ¿Usar Merge? |
|---------------------------|---------------|---------------|
| Actualizar tu rama        | ✅ Sí         | 🚫 No         |
| Subir cambios al proyecto | 🚫 No         | ✅ Sí (via PR) |

> 🔥 *Nunca hagas `merge main` dentro de tu rama de trabajo repetidamente. Usa `rebase` para mantener tu rama limpia y ordenada.*

---

## 📦 Buenas prácticas adicionales

- Commits descriptivos (`git commit -m "fix: arreglo error en login"`)
- No hagas `force push` a menos que sepas lo que haces
- Prefiere `Squash and Merge` al aprobar PRs (para historial más limpio)
- Revisa los cambios de otros antes de aprobar

---

### 📄 Ejemplo de nombres de ramas

| Tipo     | Prefijo     | Ejemplo                  |
|----------|-------------|--------------------------|
| Feature  | `feat/`     | `feat/agregar-buscador`  |
| Bugfix   | `fix/`      | `fix/error-validacion`   |
| Refactor | `refactor/` | `refactor/modulo-auth`   |
| Hotfix   | `hotfix/`   | `hotfix/crash-app`       |

---

> ✨ **Este flujo ayuda a evitar conflictos, mantener un historial limpio y facilitar revisiones entre el equipo.**
