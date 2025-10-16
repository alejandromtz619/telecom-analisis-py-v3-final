# src/aspect_rules.py
import re

# Reglas simples ES/GN/PT (las puedes ampliar)
ASPECT_PATTERNS = {
    "atencion":     r"\b(atencion|soporte|call ?center|servicio al cliente|colaborador(?:es)?)\b",
    "cobertura":    r"\b(cobertura|señal|sen(?:al|y)l(?:es)?|antena|zona sin cobertura|no hay señal)\b",
    "precio":       r"\b(precio|caro|barato|factura|facturacion|costo|tarifa|cuota)\b",
    "planes":       r"\b(plan(?:es)?|paquete(?:s)?|promo(?:cion)?(?:es)?|portabilidad)\b",
    "velocidad":    r"\b(velocidad|mbps|kbps|descarga|upload|bajada|subida|latencia|ping|lag|lento|rápido|rapido)\b",
    "instalacion":  r"\b(instalacion|instalación|tecnico|técnico|agendamiento|visita|router|modem|módem|fibra)\b",
    "fallas":       r"\b(caida|caída|corte|intermitencia|falla(?:s)?|problema|sin internet|no funciona|avería|averia)\b",
}

ASPECT_RES = {k: re.compile(v, re.I) for k, v in ASPECT_PATTERNS.items()}

def detect_aspects(text: str) -> str:
    if not isinstance(text, str):
        return ""
    found = [k for k, rx in ASPECT_RES.items() if rx.search(text)]
    return ";".join(sorted(set(found)))
