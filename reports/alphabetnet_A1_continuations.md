# Dataset de Continuations - Reporte

Generado el: 2025-11-12 21:11:37

## 1. Resumen General

- **Total de cadenas positivas procesadas:** 96,474
- **Total de autómatas únicos:** 2,986
- **Total de prefijos generados:** 60,786
- **Promedio de prefijos por autómata:** 20.36

## 2. Prefijos por Autómata

- **Autómatas con prefijo <EPS>:** 2,958 (99.06%)

| Estadística | Valor |
|------------|-------|
| Mínimo | 0 |
| Máximo | 138 |
| Media | 20.36 |
| Mediana | 10.00 |

## 3. Distribución de Positivos por Prefijo

| Estadística | Valor |
|------------|-------|
| Mínimo | 2 |
| Máximo | 2016 |
| Media | 17.12 |
| Mediana | 9.00 |
| Percentil 95 | 40.00 |
| Percentil 99 | 50.00 |

## 4. Top 20 Prefijos por Frecuencia

| Prefijo | Frecuencia Total |
|---------|------------------|
| `<EPS>` | 95,557 |
| `HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH` | 25,568 |
| `IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII` | 21,291 |
| `GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG` | 17,717 |
| `FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF` | 17,683 |
| `JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ` | 17,633 |
| `EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE` | 15,614 |
| `DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD` | 15,461 |
| `AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA` | 13,947 |
| `CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC` | 13,908 |
| `LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL` | 13,735 |
| `A` | 11,974 |
| `BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB` | 10,158 |
| `B` | 9,370 |
| `C` | 8,634 |
| `D` | 8,159 |
| `KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK` | 8,120 |
| `E` | 6,974 |
| `F` | 6,681 |
| `H` | 6,580 |

## 5. Distribución de Longitudes de Prefijos

| Estadística | Valor |
|------------|-------|
| Mínimo | 1 |
| Máximo | 64 |
| Media | 11.18 |
| Mediana | 4.00 |
| Percentil 95 | 51.00 |
| Percentil 99 | 62.00 |

✅ **El percentil 95 es 51.00, que está por debajo de max_len=64.**

## 6. Formato Ancho (Multi-hot)

- **Total de filas:** 60,786
- **Autómatas únicos:** 2,958
- **Prefijos únicos:** 3,131

## 7. Formato Largo (Binario)

- **Total de filas:** 2,081,494
- **Filas positivas:** 1,040,747
- **Filas negativas:** 1,040,747
- **Ratio neg:pos:** 1.00
- **Autómatas únicos:** 2,958
- **Prefijos únicos:** 3,131

✅ **El ratio está dentro del ±10% del esperado (1.00).**

## 8. Criterios de Aceptación

1. **Cada autómata tiene ≥ 20 prefijos:**
   - Autómatas con ≥ 20 prefijos: 629 (21.06%)
   ⚠️ **No cumplido completamente**

2. **No hay símbolos fuera del vocabulario:**
   ✅ **Verificado durante la construcción**

3. **Ratio pos:neg se respeta (±10%):**
   ✅ **Cumplido (ratio: 1.00)**

4. **p95 de longitud de prefijo ≤ max_len:**
   ✅ **Cumplido (p95: 51.00 ≤ 64)**
