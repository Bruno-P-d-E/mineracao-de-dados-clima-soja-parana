#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT EXECUTIVO: Atualiza Pipeline Completo
=============================================
Executa em sequência:
  1. Carrega e processa PAM com correção IPCA
  2. Faz merge com dados climáticos (NASA_POWER) e geográficos (IBGE)
  3. Salva dataset_final.csv e dataset_final.parquet
  4. Dashboard fica pronto para usar os dados corrigidos
"""

import subprocess
import sys
from pathlib import Path

def executar_comando(descricao, comando):
    """Executa um comando e exibe resultado."""
    print("\n" + "="*70)
    print(f"▶ {descricao}")
    print("="*70)
    try:
        resultado = subprocess.run(comando, shell=True, cwd=Path(__file__).parent)
        if resultado.returncode == 0:
            print(f"✓ {descricao} - SUCESSO\n")
            return True
        else:
            print(f"✗ {descricao} - FALHA\n")
            return False
    except Exception as e:
        print(f"✗ Erro ao executar: {e}\n")
        return False

def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         ATUALIZAÇÃO COMPLETA: PIPELINE + DASHBOARD                ║")
    print("║                  (Dados com Correção IPCA 2024)                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # ─── ETAPA 1: Merge (integra PAM + NASA + IBGE com correção IPCA) ────────
    sucesso_merge = executar_comando(
        "ETAPA 1: Executar Pipeline de Merge (merge.py)",
        "python src/preprocessing/merge.py"
    )
    
    if not sucesso_merge:
        print("❌ Pipeline falhou. Verifique os erros acima.")
        sys.exit(1)
    
    # ─── RESUMO FINAL ──────────────────────────────────────────────────────
    print("\n" + "╔════════════════════════════════════════════════════════════════════╗")
    print("║                        ✓ ATUALIZAÇÃO COMPLETA                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    print("\n📊 ARQUIVOS GERADOS:")
    print("  ✓ data/processed/dataset_final.csv       (~2.793 linhas × ~865 colunas)")
    print("  ✓ data/processed/dataset_final.parquet   (otimizado para dashboard)")
    
    print("\n✨ NOVAS COLUNAS DISPONÍVEIS:")
    print("  • Fator_Correcao  → fator IPCA aplicado")
    print("  • Valor_Corrigido → valor em R$ base 2024")
    print("  • Valor_por_Ha    → rentabilidade por hectare")
    
    print("\n🚀 PRÓXIMA ETAPA:")
    print("  Execute o dashboard com:")
    print("     streamlit run src/dashboard/dashboard.py")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
