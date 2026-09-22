"""PancakeSwap Live Prediction Executor powered by FlyDeck Agent.

Substitui a ingestão anterior baseada em mensagens do Telegram (Telethon/LSTM)
por previsões neurais diretas do FlyDeck Agent em tempo real:
- Sincronização automática com a fronteira de candles de 5 minutos da Binance.
- Análise neural completa: Retina 2D, MaleCNS (T4/T5), LPTC, Central Complex,
  Mushroom Body (memória associativa esparsa) e Predictive Coding.
- Execução automatizada no PancakeSwap Prediction via pyautogui / MSS.
- Aprendizado contínuo causal em t+1 com salvamento automático de checkpoints.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import pyautogui
except ImportError:
    pyautogui = None

# Importações dos módulos do navegador (fornecidos pelo seu projeto de automação)
try:
    from navegador_login import (
        encontrar_e_clicar_com_mss,
        encontrar_imagem,
        encontrar_qualquer_imagem,
    )
    from navegador import abrir_navegador, get_bnb_usd_price
except ImportError:
    # Mocks para execução quando os módulos gráficos estiverem em outra pasta
    print("[Aviso] Módulos 'navegador' ou 'navegador_login' não encontrados no diretório atual.")
    print("[Aviso] As ações de clique serão exibidas no console (modo simulação).")

    def encontrar_e_clicar_com_mss(caminho, threshold=0.7):
        print(f"  [Simulação Clique] Clicar em: {caminho}")
        return True

    def encontrar_imagem(caminho, threshold=0.7):
        return True

    def encontrar_qualquer_imagem(lista, threshold=0.7, popup=False):
        print(f"  [Simulação Imagem] Encontrada: {lista[0]}")
        return True

    def abrir_navegador(url):
        print(f"  [Simulação Navegador] Abrindo: {url}")

    def get_bnb_usd_price():
        return 770.0


# Importações do FlyDeck Agent
from flydeck.live_daemon import FlyDeckLiveDaemon
from flydeck.visual_agent import FlyVisualPredictionAgent
from flydeck.visual_circuit import VisualCircuit


# ==============================================================================
# CONFIGURAÇÃO DE EXECUÇÃO:
# - True:  MODO REAL (executa cliques e ordens no site do PancakeSwap)
# - False: MODO TREINO (prevê, aprende, atualiza checkpoints, SEM clicar no site)
# ==============================================================================
EXECUTAR_NO_PANCAKESWAP = True


CSV_LOG_FILE = "testes_flydeck_pancake.csv"


def salvar_previsao(dados: dict[str, Any]) -> None:
    """Salva a previsão e o estado neural no arquivo CSV."""
    file_exists = os.path.isfile(CSV_LOG_FILE)
    fieldnames = [
        "round",
        "datetime_utc",
        "close",
        "action",
        "reason",
        "confidence",
        "conflict",
        "mb_valence",
        "arousal",
        "prev_return",
    ]

    with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "round": dados.get("round", ""),
            "datetime_utc": dados.get("datetime_utc", ""),
            "close": dados.get("close", ""),
            "action": dados.get("action", ""),
            "reason": dados.get("reason", ""),
            "confidence": dados.get("confidence", ""),
            "conflict": dados.get("conflict", ""),
            "mb_valence": dados.get("mb_valence", ""),
            "arousal": dados.get("arousal", ""),
            "prev_return": dados.get("observed_return_prev", ""),
        })


def up_or_down(direcao: str, valor_aposta: str = "0.00120") -> None:
    """Executa a ordem física no site do PancakeSwap via interface gráfica."""
    if direcao == "UP":
        print(f"[{time.strftime('%H:%M:%S')}] Executando ordem: UP no PancakeSwap...")
        encontrar_e_clicar_com_mss("./imgs/up.png", threshold=0.7)
        time.sleep(2)
        encontrar_e_clicar_com_mss("./imgs/digitar_valor.png", threshold=0.7)
        time.sleep(1)
        if pyautogui is not None:
            pyautogui.write(valor_aposta, interval=0.2)
        else:
            print(f"  [Simulação] Digitando valor da aposta: {valor_aposta}")
        encontrar_e_clicar_com_mss("./imgs/confirm.png", threshold=0.7)
        time.sleep(3)
        encontrar_qualquer_imagem(["./imgs/metamask_confirm1.png", "./imgs/metamask_confirm2.png","./imgs/metamask_confirm3.png"], threshold=0.9)
        time.sleep(2)
        encontrar_qualquer_imagem(["./imgs/metamask_close2.png"], threshold=0.7)
        print(f"[{time.strftime('%H:%M:%S')}] Ordem UP enviada com sucesso!")

    elif direcao == "DOWN":
        print(f"[{time.strftime('%H:%M:%S')}] Executando ordem: DOWN no PancakeSwap...")
        encontrar_e_clicar_com_mss("./imgs/down.png", threshold=0.7)
        time.sleep(2)
        encontrar_e_clicar_com_mss("./imgs/digitar_valor.png", threshold=0.7)
        time.sleep(1)
        if pyautogui is not None:
            pyautogui.write(valor_aposta, interval=0.2)
        else:
            print(f"  [Simulação] Digitando valor da aposta: {valor_aposta}")
        encontrar_e_clicar_com_mss("./imgs/confirm.png", threshold=0.7)
        time.sleep(3)
        encontrar_qualquer_imagem(["./imgs/metamask_confirm1.png", "./imgs/metamask_confirm2.png","./imgs/metamask_confirm3.png"], threshold=0.9)
        time.sleep(2)
        encontrar_qualquer_imagem(["./imgs/metamask_close2.png"], threshold=0.7)
        print(f"[{time.strftime('%H:%M:%S')}] Ordem DOWN enviada com sucesso!")

    elif direcao == "WAIT":
        print(f"[{time.strftime('%H:%M:%S')}] Decisão WAIT: Sinal filtrado pelos circuitos neurais. Nenhuma aposta realizada (Capital Protegido).")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Direção desconhecida: {direcao}")


def logar_navegador() -> None:
    """Conecta a MetaMask no PancakeSwap."""
    print("Conectando carteira no PancakeSwap...")
    encontrar_e_clicar_com_mss("./imgs/connect_wallet.png", threshold=0.7)
    time.sleep(1)
    encontrar_e_clicar_com_mss("./imgs/meta_mask.png", threshold=0.7)
    time.sleep(1)
    encontrar_e_clicar_com_mss("./imgs/passw.png", threshold=0.7)
    time.sleep(1)
    # Insira a senha se necessário:
    # pyautogui.write('SuaSenhaMetaMask', interval=0.5)
    encontrar_e_clicar_com_mss("./imgs/logar1.png", threshold=0.7)
    time.sleep(1)


class PancakeFlyDeckBridge:
    """Ponte entre as decisões neurais do FlyDeck e as ações no PancakeSwap."""

    def __init__(
        self,
        circuit_path: str = "data/malecns/motion_visual.json",
        confidence_threshold: float = 0.15,
        valor_aposta: str = "0.00480",
        max_execucoes_por_hora: int = 3,
        lead_time_seconds: float = 25.0,
        executar_pancake: bool = EXECUTAR_NO_PANCAKESWAP,
    ) -> None:
        self.valor_aposta = valor_aposta
        self.max_execucoes_por_hora = max_execucoes_por_hora
        self.lead_time_seconds = lead_time_seconds
        self.executar_pancake = executar_pancake
        self.historico_execucoes: dict[str, int] = {}

        print("[FlyDeck Bridge] Inicializando o circuito visual MaleCNS v1.0...")
        circuit = VisualCircuit.load(circuit_path)
        self.agent = FlyVisualPredictionAgent(
            circuit,
            retina_width=32,
            retina_height=16,
            confidence_threshold=confidence_threshold,
        )

        self.daemon = FlyDeckLiveDaemon(
            agent=self.agent,
            symbol="BNBUSDT",
            interval="5m",
            checkpoint_file="data/checkpoints/live_fly_brain.json",
            diagnostics_file="data/logs/live_diagnostics.jsonl",
            lead_time_seconds=self.lead_time_seconds,
            on_action=self.ao_receber_previsao,
        )

    def ao_receber_previsao(self, action: str, record: dict[str, Any]) -> None:
        """Callback acionado automaticamente a cada 5 minutos (T - lead_time) antes do lock."""
        hora_atual = time.strftime("%H", time.localtime())
        if hora_atual not in self.historico_execucoes:
            self.historico_execucoes[hora_atual] = 0

        salvar_previsao(record)

        if action in ("UP", "DOWN"):
            # Trava de segurança por hora (se configurada)
            if self.historico_execucoes[hora_atual] >= self.max_execucoes_por_hora:
                print(f"[{time.strftime('%H:%M:%S')}] Limite de {self.max_execucoes_por_hora} operações atingido na hora {hora_atual}. Aguardando...")
                return

            print(f"\n=======================================================")
            print(f"🎯 OPORTUNIDADE IDENTIFICADA PELO FLYDECK: {action}")
            print(f"   Confiança: {record['confidence']:.2f} | Motivo: {record['reason']}")
            print(f"   Valência MB: {record['mb_valence']:.2f} | Conflito: {record['conflict']:.2f}")
            print(f"=======================================================")

            if self.executar_pancake:
                up_or_down(action, valor_aposta=self.valor_aposta)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 🧪 [MODO TREINO] Ordem {action} registrada (nenhuma ação enviada ao site).")

            self.historico_execucoes[hora_atual] += 1
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Agente optou por WAIT ({record['reason']}) | Conflito: {record['conflict']:.2f}")

    def iniciar(self) -> None:
        """Inicia a execução contínua sincronizada com a Binance."""
        modo_str = "🟢 MODO REAL (Ordens enviadas ao PancakeSwap)" if self.executar_pancake else "🧪 MODO TREINO / SIMULAÇÃO (Sem cliques no site)"
        print(f"\n[FlyDeck Bridge] Status: {modo_str}")
        print(f"[FlyDeck Bridge] Iniciando daemon (antecedência: {self.lead_time_seconds:.0f}s antes do lock)...")
        self.daemon.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PancakeSwap Live Prediction Executor (FlyDeck Agent)")
    parser.add_argument("--circuit", default="data/malecns/motion_visual.json", help="Caminho para o circuito MaleCNS JSON")
    parser.add_argument("--confidence", type=float, default=0.15, help="Limiar de confiança base do FlyDeck")
    parser.add_argument("--stake", default="0.00480", help="Valor em BNB por aposta (ex: 0.00480)")
    parser.add_argument("--max-hourly", type=int, default=6, help="Máximo de operações permitidas por hora")
    parser.add_argument("--lead-time", type=float, default=25.0, help="Segundos de antecedência antes da virada do candle (padrão: 25.0s)")
    parser.add_argument("--treino", "--simular", dest="treino", action="store_true", help="Ativa modo treino: prevê e aprende sem clicar no PancakeSwap")
    parser.add_argument("--real", dest="real", action="store_true", help="Força modo real com apostas no PancakeSwap")
    parser.add_argument("--open-browser", action="store_true", help="Abre o navegador na URL do PancakeSwap ao iniciar")
    parser.add_argument("--login", action="store_true", help="Tenta logar a MetaMask ao iniciar")
    parser.add_argument("--test-image", type=str, default=None, help="Testa detecção e clique em uma imagem específica (ex: up.png)")
    parser.add_argument("--test-scan", action="store_true", help="Escaneia a tela atual e exibe a confiança de todas as imagens")
    args = parser.parse_args()

    if args.test_scan or args.test_image:
        from test_image_click import escanear_tela, testar_clique_imagem
        if args.test_scan:
            escanear_tela()
        elif args.test_image:
            testar_clique_imagem(args.test_image, dry_run=False)
        sys.exit(0)

    # Determina o modo de execução: CLI tem precedência sobre a variável do arquivo
    executar_no_site = EXECUTAR_NO_PANCAKESWAP
    if args.treino:
        executar_no_site = False
    elif args.real:
        executar_no_site = True

    if executar_no_site:
        if args.open_browser:
            print("[Início] Abrindo o navegador...")
            abrir_navegador("https://pancakeswap.finance/prediction?token=BNB")
            time.sleep(4)

        if args.login:
            logar_navegador()
    else:
        print("\n" + "=" * 65)
        print("🧪 [MODO TREINO ATIVADO]")
        print("O agente irá executar 24/7, realizar previsões, treinar o Mushroom Body,")
        print("salvar checkpoints e registrar logs no CSV, mas NÃO interagirá com o site.")
        print("=" * 65)

    bridge = PancakeFlyDeckBridge(
        circuit_path=args.circuit,
        confidence_threshold=args.confidence,
        valor_aposta=args.stake,
        max_execucoes_por_hora=args.max_hourly,
        lead_time_seconds=args.lead_time,
        executar_pancake=executar_no_site,
    )
    bridge.iniciar()
