# Proyecto 1: Organización e Indexación Eficiente de Archivos con Datos Multidimensionales

**Curso:** CS2702 – Base de Datos 2  
**Universidad:** UTEC  
**Integrantes:**  
- José Armando Arias Romero  
- Diego Illescas
- Jose Ignacio Aguilar Millones
- Fabricio Bautista

**Repositorio:** [\[enlace al GitHub\] ](https://github.com/j4guil4r/proyecto_base2) 

---

## 1. Introducción

### 1.1 Objetivo del Proyecto
El objetivo principal del proyecto es implementar un gestor de bases de datos que permita la organización eficiente de archivos en memoria secundaria, utilizando técnicas de indexación y estructuras de datos para optimizar las operaciones de inserción, búsqueda y eliminación.

### 1.2 Aplicación del Proyecto
Los índices empleados en este proyecto pueden tener diversas aplicaciones, detallamos algunas a continuación:

- Sistema de gestión de libros electrónicos:
    - B+ Tree: Para buscar libros por autor, título, fecha de publicación o por género.

    - Extensible Hashing: Se puede usar la llave del Libro o Autor para una búsqueda eficiente.

    - ISAM: Para manejar índices de libros que tienen una organización secuencial basada en fecha de publicación.

    - R-Tree: Para localizar libros dentro de una categoría espacial. Por ejemplo, "libros más vendidos en tu área" o basados en preferencias geográficas.

- Sistema de gestión de salud (Historia Clínica Electrónica)

    - B+ Tree: Para consultas rápidas sobre la fecha de ingreso o historial médico de un paciente.

    - Extendible Hashing: Para localizar a los pacientes rápidamente por su ID o número de seguro social.

    - ISAM: Para manejar registros médicos de forma secuencial

    - R-Tree: Para identificar a los pacientes por ubicación geográfica en caso de emergencias o para optimizar la asignación de recursos médicos cercanos, en caso que la clínica tenga varias sedes.

- Sistema de geolocalización para búsqueda de eventos

    - B+ Tree: Para búsquedas rápidas por fecha de evento o tipo, como conciertos, películas en cines, partidos de fútbol, entre otros.

    - R-Tree: Para encontrar eventos cercanos a una ubicación geográfica específica o en un área de interés.

    - Extendible Hashing: Para realizar búsquedas rápidas por código postal, distrito o ciudad.

    - ISAM: Para almacenar los eventos en un formato secuencial y optimizar la consulta de eventos por fecha o por tipo de evento.

### 1.3 Resultados Esperados
- Reducción de accesos a disco en comparación con un archivo secuencial sin índices.
- Disminución del tiempo de inserción y búsqueda en grandes volúmenes de datos.
- Flexibilidad en manejar diferentes tipos de tablas con diferentes atributos.

---

## 2. Técnicas Utilizadas

### 2.1 Descripción General

En el presente proyecto se implementaron diversas técnicas de indexación para optimizar el acceso a datos almacenados en archivos.

**Sequential File:**  
Organiza los registros de forma ordenada según una clave primaria. Las inserciones nuevas se almacenan temporalmente en un área auxiliar, y cuando se alcanza un umbral definido se reconstruye el archivo principal manteniendo el orden.

**ISAM (Sparse Index):**  
Construye un índice estático de varios niveles que apunta a bloques de datos organizados secuencialmente. Permite búsquedas rápidas mediante el índice y maneja nuevas inserciones con páginas de desbordamiento. Es eficiente para bases con pocas modificaciones estructurales.

**Extendible Hashing:**  
Emplea una función de hash dinámica que adapta su profundidad según el crecimiento de los datos. Minimiza colisiones y garantiza acceso directo a registros individuales. Ideal para búsquedas exactas por clave.

**B+ Tree:**  
Estructura balanceada en forma de árbol donde todos los valores están en las hojas, enlazadas secuencialmente. Ofrece alta eficiencia para inserciones, eliminaciones y búsquedas por rango. Es la técnica de indexación más utilizada en sistemas de bases de datos reales.

**R-Tree:**  
Índice espacial que organiza objetos multidimensionales (como coordenadas) mediante rectángulos mínimos que se agrupan jerárquicamente. Permite búsquedas por área o por vecinos cercanos, resultando adecuado para consultas geográficas.
 

### 2.2 Algoritmos de Operaciones

#### 2.2.1 Sequential File

**Insersión**: La inserción se realiza en un archivo auxiliar para evitar reordenar todo el archivo principal en cada operación. Cada nuevo registro se empaqueta y se añade secuencialmente al archivo auxiliar (`.aux`).  
Si la cantidad de registros en este archivo auxiliar alcanza una capacidad máxima predefinida, se ejecuta la reconstrucción del archivo principal (`.dat`) con el auxiliar, ordenando todos los registros según la clave y reemplazando el archivo original.

**Pseudocódigo**:

```text
Función add(registro):
    Escribir registro en archivo_auxiliar
    Si tamaño(archivo_auxiliar) >= capacidad_auxiliar:
        reconstruir_archivo()

Función reconstruir_archivo():
    Leer registros_main desde archivo_principal
    Leer registros_aux desde archivo_auxiliar
    Unir todos los registros
    Ordenar por clave
    Sobrescribir archivo_principal con los registros ordenados
    Vaciar archivo_auxiliar
```
---

**Eliminación**: La eliminación implica una reconstrucción costosa, ya que el archivo secuencial no admite eliminaciones directas sin romper el orden físico.  
El algoritmo combina los registros del archivo principal y auxiliar, filtra aquellos cuya clave coincide con la que se desea eliminar, ordena los registros restantes y sobrescribe el archivo principal con el nuevo conjunto limpio. Finalmente, el archivo auxiliar se vacía.  

**Pseudocódigo:**
```text
Función remove(clave):
    Leer registros desde archivo_principal y archivo_auxiliar
    Mantener solo los registros cuya clave ≠ clave
    Ordenar los registros restantes por clave
    Sobrescribir archivo_principal con los registros restantes
    Vaciar archivo_auxiliar
```
---
**Búsqueda**: 
Primero se recorren los registros del archivo auxiliar buscando coincidencias exactas. Luego, mediante una búsqueda binaria, se localiza en el archivo principal la posición del primer registro con clave mayor o igual a la buscada. A partir de esa posición, se leen los registros secuenciales hasta que la clave del registro sea mayor que la buscada.  
Ambos resultados se combinan y devuelven al usuario.  

**Pseudocódigo:**

```text
Función search(clave):
    resultados = []

    Para cada registro en archivo_auxiliar:
        Si registro.clave == clave:
            Agregar registro a resultados

    índice = búsqueda_binaria_en_archivo_principal(clave)

    Para i desde índice hasta fin de archivo_principal:
        registro = leer_registro(i)
        Si registro.clave == clave:
            Agregar registro a resultados
        Sino si registro.clave > clave:
            Romper ciclo

    Return resultados
```
---
**Búsqueda por rango**: 
Se recorren los registros del archivo auxiliar y principal, filtrando los que cumplen con el rango y ordenándolos al final antes de devolverlos.

**Pseudocódigo**
```text
Función rangeSearch(inicio, fin):
    resultados = []

    Para cada registro en archivo_auxiliar:
        Si inicio <= registro.clave <= fin:
            Agregar registro a resultados

    índice ← búsqueda_binaria_en_archivo_principal(inicio)

    Para i desde índice hasta fin de archivo_principal:
        registro = leer_registro(i)
        Si registro.clave <= fin:
            Agregar registro a resultados
        Sino:
            Romper ciclo

    Ordenar resultados por clave
    Return resultados
```
#### 2.2.2 ISAM (Sparse Index)
**Inserción**
Con la clave, se navega por el índice L2, es decir, el superior y luego por el índice L1 (inferior) hasta localizar la página de datos objetivo. Se recorre la cadena de desbordamiento hasta la última página de la cadena.  
- Si la última página tiene espacio: se inserta el registro, se ordena por clave dentro de la página y se reescribe el bloque.  
- Si está llena: se crea nueva página de overflow con el registro, se enlaza desde la página anterior y se persiste.

**Pseudocódigo:**

```text
función add(clave, RID):
    nodo_L2 ← leer_raíz_L2()
    off_L1 ← puntero_menor_o_igual(nodo_L2, clave)
    si off_L1 es nulo: error

    nodo_L1 ← leer_nodo_L1(off_L1)
    off_data ← puntero_menor_o_igual(nodo_L1, clave)
    si off_data es nulo: error

    archivo_actual ← DAT
    off_actual ← off_data
    página ← leer_página(archivo_actual, off_actual)

    // avanzar al final de la cadena de overflow
    mientras página.next_overflow_offset ≠ -1:
        archivo_actual ← OVF
        off_actual ← página.next_overflow_offset
        página ← leer_página(archivo_actual, off_actual)

    si página.tiene_espacio():
        página.registros.agregar((clave, RID, activo=true))
        ordenar(página.registros por clave)
        escribir_página(archivo_actual, off_actual, página)
    sino:
        nueva_página ← página_vacía()
        nueva_página.registros.agregar((clave, RID, activo=true))
        off_nueva ← escribir_página(OVF, fin, nueva_página)

        página.next_overflow_offset ← off_nueva
        escribir_página(archivo_actual, off_actual, página)
```

---

**Búsqueda exacta**: Se navega desde L2 a L1 y a la página de datos. Luego se escanea la página objetivo y su overflow, devolviendo todos los RIDs cuya clave coincide y estén activos.

**Pseudocódigo:**
```text
función search(clave):
    resultados ← []

    nodo_L2 ← leer_raíz_L2()
    off_L1 ← puntero_menor_o_igual(nodo_L2, clave)
    si off_L1 es nulo: devolver []

    nodo_L1 ← leer_nodo_L1(off_L1)
    off_data ← puntero_menor_o_igual(nodo_L1, clave)
    si off_data es nulo: devolver []

    archivo_actual ← DAT
    off_actual ← off_data

    mientras off_actual ≠ -1:
        página ← leer_página(archivo_actual, off_actual)

        para cada (k, rid, activo) en página.registros:
            si activo y k == clave:
                resultados.agregar(rid)

        off_actual ← página.next_overflow_offset
        archivo_actual ← OVF  // a partir de aquí, lectura en overflow

    devolver resultados
```

---

**Eliminación**: Se navega por los niveles a la página de datos, y luego se recorre la página y su cadena de overflow marcando como inactivos los registros con la llave a eliminar. 
Se incluye un corte temprano si el máximo de la página actual es menor que la clave buscada.

**Pseudocódigo:**
```text
función remove(clave):
    nodo_L2 ← leer_raíz_L2()
    off_L1 ← puntero_menor_o_igual(nodo_L2, clave)
    si off_L1 es nulo: retornar

    nodo_L1 ← leer_nodo_L1(off_L1)
    off_data ← puntero_menor_o_igual(nodo_L1, clave)
    si off_data es nulo: retornar

    archivo_actual ← DAT
    off_actual ← off_data

    mientras off_actual ≠ -1:
        página ← leer_página(archivo_actual, off_actual)
        mod ← falso

        para i desde 0 hasta len(página.registros)-1:
            (k, rid, activo) ← página.registros[i]
            si activo y k == clave:
                página.registros[i] ← (k, rid, falso)
                mod ← verdadero

        si mod:
            escribir_página(archivo_actual, off_actual, página)

        // si la página está ordenada y su mayor clave < clave, no hay más coincidencias
        si no vacía(página.registros) y mayor_clave(página) < clave:
            romper

        off_actual ← página.next_overflow_offset
        archivo_actual ← OVF
```

---

**Búsqueda por rango**: Se ubica la página de inicio usando L2 -> L1 y el vector `data_page_offsets`. A partir de esa página, se escanean en orden las páginas principales y sus overflows, acumulando RIDs con `start_key <= k <= end_key`. Si en una página principal se encuentra una clave `k > end_key`, se detiene la exploración hacia adelante.

**Pseudocódigo:**
```text
función rangeSearch(inicio, fin):
    resultados ← []

    nodo_L2 ← leer_raíz_L2()
    off_L1 ← puntero_menor_o_igual(nodo_L2, inicio)
    si off_L1 es nulo: devolver []

    nodo_L1 ← leer_nodo_L1(off_L1)
    off_inicio ← puntero_menor_o_igual(nodo_L1, inicio)
    si off_inicio es nulo: devolver []

    idx ← índice_de(off_inicio en data_page_offsets)
    si idx no existe: devolver []

    detener ← falso
    para i desde idx hasta fin_de(data_page_offsets):
        si detener: romper

        archivo_actual ← DAT
        off_actual ← data_page_offsets[i]

        mientras off_actual ≠ -1:
            página ← leer_página(archivo_actual, off_actual)

            para cada (k, rid, activo) en página.registros:
                si no activo: continuar

                si inicio ≤ k ≤ fin:
                    resultados.agregar(rid)

                si k > fin:
                    si archivo_actual == DAT:
                        detener ← verdadero
                        romper
                    sino:
                        off_actual ← -1
                        romper

            si off_actual ≠ -1:
                off_actual ← página.next_overflow_offset
                archivo_actual ← OVF

    devolver resultados
```

---

---
#### 2.2.3 Extendible Hashing

**Inserción**: Se calcula el índice del bucket usando `global_depth`.  
- Si el bucket tiene espacio, se añade `(key, RID)`.  
- Si está lleno, se divide el bucket: se incrementa `local_depth`, se redistribuyen los pares según los nuevos bits, se actualizan los punteros del directorio y, si hace falta, se duplica el directorio. Luego se reintenta la inserción.

**Pseudocódigo**
```text
función ADD(key, rid):
    idx ← INDEX_FOR(key, global_depth) // hash(key) & ((1 << global_depth) - 1)
    off ← bucket_pointers[idx]
    B ← READ_BUCKET(off)

    si NOT FULL(B) entonces
        APPEND(B.values, (key, rid))
        WRITE_BUCKET(off, B)
        retornar

    SPLIT_BUCKET(idx, B, off)
    // reintentar luego del split (tail recursion)
    ADD(key, rid)

```

---
**Búsqueda exacta**: Se calcula el índice del bucket con `global_depth`, se lee el bucket y se devuelven los RIDs de las tuplas cuya clave coincide exactamente.

**Pseudocódigo**
```text
función SEARCH(key) → Lista<RID>:
    idx ← INDEX_FOR(key, global_depth)
    off ← bucket_pointers[idx]
    B ← READ_BUCKET(off)

    resultados ← []
    para cada (k, rid) en B.values hacer
        si k = key entonces
            APPEND(resultados, rid)
    retornar resultados
```
**Eliminación**: Se localiza el bucket y se filtran las tuplas a eliminar.  
En esta versión, no se realiza **merge** de buckets ni decrecimiento del directorio; solo se compacta el contenido del bucket en disco.

**Pseudocódigo**
```text
función REMOVE(key, rid_opcional):
    idx ← INDEX_FOR(key, global_depth)
    off ← bucket_pointers[idx]
    B ← READ_BUCKET(off)

    si rid_opcional está definido entonces
        B.values ← [ (k, r) en B.values | (k, r) ≠ (key, rid_opcional) ]
    sino
        B.values ← [ (k, r) en B.values | k ≠ key ]

    WRITE_BUCKET(off, B)
```

#### 2.2.4 B+ Tree

**Inserción**: Se localiza la hoja con `_find_leaf(key)`, se inserta ordenadamente la `(key, value)`; si la hoja queda llena, se divide en dos hojas y se inserta la clave promovida en el padre. Si el padre se llena, también se divide recursivamente; si no hay padre, se crea una nueva raíz.

**Pseudocódigo**
```text
procedimiento INSERT(key, rid):
    leaf ← FIND_LEAF(key)
    LEAF_INSERT(leaf, key, rid)

    si IS_FULL(leaf) entonces
        SPLIT_LEAF(leaf)
    sino
        WRITE_NODE(leaf)

```

---

**Búsqueda exacta**: Se desciende desde la raíz hasta la hoja que podría contener la clave, usando `bisect_right` en nodos internos. Si la clave está en la hoja, se devuelven sus RIDs; si no, se devuelve vacío.

**Pseudocódigo**
``` text
función SEARCH(key) → Lista<RID>:
    leaf ← FIND_LEAF(key)
    i ← INDEX_OF(leaf.keys, key) // devuelve error si no existe
    si i existe entonces
        return leaf.values[i]
    sino
        return []
```
---
**Búsqueda por rango**: Se encuentra la hoja de inicio con `start_key` y se recorren hojas enlazadas vía `next_leaf`, acumulando RIDs para todas las claves entre `start_key` y `end_key`. Se detiene cuando se supera `end_key`.

**Pseudocódigo**
```text
función RANGE_SEARCH(start_key, end_key) → Lista<RID>:
    results ← []
    leaf ← FIND_LEAF(start_key)

    mientras leaf ≠ NULL hacer
        para i ← 0 hasta |leaf.keys| - 1 hacer
            k ← leaf.keys[i]
            si start_key ≤ k ≤ end_key entonces
                EXTEND(results, leaf.values[i])
            sino si k > end_key entonces
                return results

        si leaf.next_leaf ≠ -1 entonces
            leaf ← READ_NODE(leaf.next_leaf)
        sino
            leaf ← NULL

    return results
```
---

**Eliminación**: Se busca la hoja. Si se pasa `value`, se elimina solo ese RID de la lista; si queda vacía, se borra la clave. Si no se pasa `value`, se borra toda la clave. Luego se escribe la hoja. No se implementa merge/redistribución, por lo que el árbol puede quedar subutilizado tras muchas bajas.

**Pseudocódigo**
```text
funcion REMOVE(key, value_opcional):
    leaf ← FIND_LEAF(key)
    i ← INDEX_OF(leaf.keys, key)
    si i no existe entonces
        retornar

    si value_opcional existe entonces
        REMOVE_ONE(leaf.values[i], value_opcional)
        si |leaf.values[i]| = 0 entonces
            ERASE_AT(leaf.keys, i)
            ERASE_AT(leaf.values, i)
    sino
        ERASE_AT(leaf.keys, i)
        ERASE_AT(leaf.values, i)


    WRITE_NODE(leaf)
```
---

#### 2.2.5 R-Tree

**Inserción**: Convertimos el punto en un MBR degenerado y lo insertamos con su `RID`. La librería se encarga de elegir la hoja, ajustar MBRs y dividir nodos si es necesario.

**Pseudocódigo**
```text
función ADD(point, rid):
    // point: (p1, p2, ..., pd)
    bbox ← (p1, p1, p2, p2, ..., pd, pd) // min=max por dimensión
    RTREE.INSERT(rid, bbox)
```
---

**Eliminación**:  Para borrar en un R-Tree hay que especificar el RID y el mismo MBR con el que se insertó.  
Dado que representamos puntos como cajas degeneradas, reconstruimos el mismo `bbox` y pedimos a la librería que elimine esa entrada exacta.

**Pseudocódigo**
```text
función REMOVE(point, rid):
    si rid es nulo entonces
    error "Se requiere el RID para eliminar"

    bbox ← (p1, p1, p2, p2, ..., pd, pd)
    RTREE.DELETE(rid, bbox)
```


---

**Búsqueda por radio**: Aproximamos la circunferencia/hiperesfera por un hiper-rectángulo centrado en `point` con semiejes `radius`. Consultamos por intersección con ese rectángulo y recuperamos los RIDs candidatos.  

**Pseudocódigo**
```text
función RADIUS_SEARCH(point, radius) → Lista<RID>:
    // Construir caja: [p1-r, p1+r, p2-r, p2+r, ..., pd-r, pd+r]
    min_coords ← [pi - radius para cada pi en point]
    max_coords ← [pi + radius para cada pi en point]
    query_box ← CONCAT(min_coords, max_coords)
    resultados ← RTREE.INTERSECTION(query_box)   // IDs que intersectan la caja
    retornar LISTA(resultados)
```
---

**Búsqueda de k vecinos más cercanos**: La librería implementa KNN sobre el índice. Le pasamos el punto de consulta y `k`, y retorna los `k` RIDs más cercanos según la métrica por defecto (generalmente euclidiana en Rtree).

**Pseudocódigo**
```text
función KNN_SEARCH(point, k) → Lista<RID>:
    resultados ← RTREE.NEAREST(point, k)
    retornar LISTA(resultados)
```
---


### 2.3 Análisis comparativo teórico

### 2.4 Parser SQL
El parser SQL se desarrolló como un analizador sintáctico ligero construido sobre expresiones regulares de Python (re). Interpreta comandos SQL básicos y los transforma en un diccionario que el gestor pueda leer.
Utiliza la librería estándar re, la cual permite reconocer y extraer patrones específicos de sentencias como CREATE TABLE, INSERT INTO, SELECT, DELETE, así como sus variantes con cláusulas FROM FILE o WHERE.

**Input**

Recibe una cadena SQL sin procesar, por ejemplo:
```sql
CREATE TABLE clientes (id INT, nombre VARCHAR[50] INDEX EHASH);
INSERT INTO clientes VALUES (1, 'Jose');
SELECT * FROM clientes WHERE id = 1;
```

**Output**

Devuelve un diccionario que describe la intención del comando, listo para usar por el motor del sistema.
Ejemplo:

```json
{
  "command": "SELECT",
  "table_name": "clientes",
  "where": {"column": "id", "op": "=", "value": 1}
}
````

## 3. Resultados Experimentales

## 4. Pruebas de uso

### 4.1 Pruebas de la interfaz

### 4.2 Video
Enlace del video :