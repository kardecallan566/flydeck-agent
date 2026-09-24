"""Utilitário para Teste Manual de Detecção e Clique em Imagens (PancakeSwap / MetaMask).

Permite testar manualmente:
- Clique ou movimento em uma imagem específica (ex: up.png, down.png, confirm.png).
- Varredura da tela atual (Scanner): mostra quais templates estão visíveis e a porcentagem de confiança.
- Teste de fluxos completos (UP, DOWN, Login) em modo real ou modo seguro (dry-run sem clicar).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mss
except ImportError:
    mss = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

# Informa ao pytest para não coletar este script como suíte de testes
__test__ = False


IMGS_DIR = Path("./imgs")


def contagem_regressiva(segundos: int = 3, mensagem: str = "Posicione a janela na tela...") -> None:
    """Dá tempo para o usuário alternar para o navegador."""
    print(f"\n⏳ {mensagem}")
    for i in range(segundos, 0, -1):
        print(f"   Iniciando em {i}...")
        time.sleep(1)
    print("   ▶ Começando agora!\n")


def listar_templates_disponiveis() -> list[str]:
    """Retorna lista de nomes de imagens na pasta ./imgs."""
    if not IMGS_DIR.exists():
        return []
    return sorted([f.name for f in IMGS_DIR.glob("*.png")])


def testar_clique_imagem(
    nome_imagem: str,
    threshold: float = 0.7,
    timeout: int = 15,
    dry_run: bool = False,
    delay_antes: int = 3,
) -> bool:
    """Testa a detecção e o clique em uma imagem específica."""
    if cv2 is None or mss is None or pyautogui is None:
        print("[Erro] cv2, mss ou pyautogui não estão instalados no ambiente Python.")
        return False

    caminho = IMGS_DIR / nome_imagem if not nome_imagem.startswith("./") and not os.path.isabs(nome_imagem) else Path(nome_imagem)
    if not caminho.suffix:
        caminho = caminho.with_suffix(".png")

    if not caminho.exists():
        print(f"[Erro] Arquivo não encontrado: {caminho}")
        disponiveis = listar_templates_disponiveis()
        print(f"Imagens disponíveis em {IMGS_DIR}: {', '.join(disponiveis)}")
        return False

    template = cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"[Erro] Falha ao ler a imagem com OpenCV: {caminho}")
        return False

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[1], template.shape[0]

    if delay_antes > 0:
        contagem_regressiva(delay_antes, f"Buscando por '{caminho.name}'. Alterne para a tela alvo")

    print(f"🔍 Procurando por '{caminho.name}' (threshold={threshold:.2f}, timeout={timeout}s)...")
    start_time = time.time()

    with mss.mss() as sct:
        monitor = sct.monitors[0]

        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            elapsed = time.time() - start_time
            print(f"   [{elapsed:4.1f}s] Confiança detectada: {max_val * 100:5.1f}% (alvo: {threshold * 100:.0f}%)", end="\r")

            if max_val >= threshold:
                center_x = monitor["left"] + max_loc[0] + w // 2
                center_y = monitor["top"] + max_loc[1] + h // 2
                print(f"\n\n🎯 IMAGEM ENCONTRADA com {max_val * 100:.1f}% de confiança!")
                print(f"   Localização: Centro ({center_x}, {center_y}) | Caixa: [x={max_loc[0]}, y={max_loc[1]}, w={w}, h={h}]")

                if dry_run:
                    print("   [DRY-RUN] Movendo cursor suavemente até o alvo (SEM CLICAR)...")
                    pyautogui.moveTo(center_x, center_y, duration=0.6)
                    print("   [DRY-RUN] Cursor posicionado com sucesso!")
                else:
                    print(f"   [CLIQUE] Movendo e clicando em ({center_x}, {center_y})...")
                    pyautogui.moveTo(center_x, center_y, duration=0.3)
                    time.sleep(0.3)
                    pyautogui.click()
                    print("   [CLIQUE] Clique executado com sucesso!")
                return True

            if elapsed > timeout:
                print(f"\n\n❌ [TIMEOUT] Imagem '{caminho.name}' NÃO encontrada após {timeout}s (Melhor confiança: {max_val * 100:.1f}%).")
                print("   Dicas:")
                print("   - Verifique se a janela está visível e não coberta.")
                print(f"   - Tente reduzir o threshold (ex: --threshold 0.6) se a imagem tiver mudado levemente.")
                return False

            time.sleep(0.5)


def escanear_tela(threshold: float = 0.6) -> None:
    """Captura a tela atual e testa todos os templates da pasta ./imgs, exibindo o ranking de correspondência."""
    if cv2 is None or mss is None:
        print("[Erro] cv2 ou mss não instalados.")
        return

    templates = listar_templates_disponiveis()
    if not templates:
        print(f"Nenhum template encontrado em {IMGS_DIR}")
        return

    contagem_regressiva(3, "Escaneando tela. Coloque a página do PancakeSwap / MetaMask visível")

    print(f"📸 Capturando tela e testando {len(templates)} imagens em ./imgs/...\n")

    with mss.mss() as sct:
        monitor = sct.monitors[0]
        screenshot = np.array(sct.grab(monitor))
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    resultados: list[tuple[str, float, int, int]] = []

    for t_name in templates:
        caminho = IMGS_DIR / t_name
        img = cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w, h = img.shape[1], img.shape[0]

        res = cv2.matchTemplate(screenshot_gray, gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        cx = monitor["left"] + max_loc[0] + w // 2
        cy = monitor["top"] + max_loc[1] + h // 2
        resultados.append((t_name, max_val, cx, cy))

    # Ordenar por maior confiança
    resultados.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Template':<26} | {'Confiança':<10} | {'Status':<12} | {'Posição (X, Y)':<15}")
    print("-" * 72)

    for nome, conf, x, y in resultados:
        pct = conf * 100
        if conf >= threshold:
            status = "✅ VISÍVEL"
        elif conf >= 0.5:
            status = "⚠️  PARCIAL"
        else:
            status = "❌ NÃO"

        print(f"{nome:<26} | {pct:6.2f}%   | {status:<12} | ({x:4d}, {y:4d})")
    print("-" * 72)


def testar_fluxo(fluxo: str, dry_run: bool = True) -> None:
    """Testa uma sequência de passos (ex: UP, DOWN, LOGIN)."""
    if fluxo.lower() == "up":
        passos = [
            ("up.png", 0.7, "1. Clicar no botão UP (Card NEXT)"),
            ("digitar_valor.png", 0.7, "2. Clicar no campo para digitar valor"),
            ("confirm.png", 0.7, "3. Clicar no botão de confirmação"),
        ]
    elif fluxo.lower() == "down":
        passos = [
            ("down.png", 0.7, "1. Clicar no botão DOWN (Card NEXT)"),
            ("digitar_valor.png", 0.7, "2. Clicar no campo para digitar valor"),
            ("confirm.png", 0.7, "3. Clicar no botão de confirmação"),
        ]
    elif fluxo.lower() == "login":
        passos = [
            ("connect_wallet.png", 0.7, "1. Clicar em Connect Wallet"),
            ("meta_mask.png", 0.7, "2. Escolher MetaMask"),
            ("passw.png", 0.7, "3. Campo de senha MetaMask"),
            ("logar1.png", 0.7, "4. Botão de login MetaMask"),
        ]
    else:
        print(f"Fluxo desconhecido: {fluxo}. Opções: up, down, login")
        return

    print(f"\n=======================================================")
    print(f"🚀 INICIANDO TESTE DO FLUXO: {fluxo.upper()} ({'DRY-RUN - Apenas Move Cursor' if dry_run else 'REAL - Irá Clicar!'})")
    print(f"=======================================================")

    contagem_regressiva(4, "Prepare o navegador com a tela correspondente")

    for img, thresh, desc in passos:
        print(f"\n👉 Passo: {desc}")
        sucesso = testar_clique_imagem(img, threshold=thresh, timeout=12, dry_run=dry_run, delay_antes=0)
        if not sucesso:
            print(f"\n⚠️  Fluxo interrompido no passo '{img}'.")
            return
        time.sleep(1.5)

    print(f"\n🎉 FLUXO {fluxo.upper()} CONCLUÍDO COM SUCESSO!")


def menu_interativo() -> None:
    """Menu interativo quando o script é executado sem parâmetros."""
    while True:
        print("\n" + "=" * 55)
        print("   TESTADOR MANUAL DE CLIQUE EM IMAGENS (FLYDECK)")
        print("=" * 55)
        print("1. Escanear tela atual (Ver o que está visível e a % de confiança)")
        print("2. Testar imagem específica (Move cursor / Clica)")
        print("3. Testar fluxo UP completo (up -> digitar_valor -> confirm)")
        print("4. Testar fluxo DOWN completo (down -> digitar_valor -> confirm)")
        print("5. Testar fluxo de Login (connect -> metamask -> login)")
        print("6. Listar todas as imagens disponíveis em ./imgs")
        print("0. Sair")
        print("=" * 55)

        escolha = input("Escolha uma opção (0-6): ").strip()

        if escolha == "0":
            print("Saindo...")
            break
        elif escolha == "1":
            thresh_str = input("Threshold mínimo [padrão 0.60]: ").strip()
            thresh = float(thresh_str) if thresh_str else 0.60
            escanear_tela(threshold=thresh)
        elif escolha == "2":
            disponiveis = listar_templates_disponiveis()
            print("\nImagens disponíveis:")
            for idx, nome in enumerate(disponiveis, 1):
                print(f"  {idx}. {nome}")
            escolha_img = input("\nDigite o número ou nome da imagem (ex: up.png): ").strip()
            if escolha_img.isdigit() and 1 <= int(escolha_img) <= len(disponiveis):
                nome_alvo = disponiveis[int(escolha_img) - 1]
            else:
                nome_alvo = escolha_img

            dry_input = input("Modo seguro DRY-RUN (apenas mover cursor sem clicar)? (S/N) [padrão: S]: ").strip().upper()
            dry_run = dry_input != "N"

            testar_clique_imagem(nome_alvo, threshold=0.7, timeout=15, dry_run=dry_run, delay_antes=3)
        elif escolha == "3":
            dry_input = input("Modo seguro DRY-RUN (apenas mover cursor sem clicar)? (S/N) [padrão: S]: ").strip().upper()
            testar_fluxo("up", dry_run=(dry_input != "N"))
        elif escolha == "4":
            dry_input = input("Modo seguro DRY-RUN (apenas mover cursor sem clicar)? (S/N) [padrão: S]: ").strip().upper()
            testar_fluxo("down", dry_run=(dry_input != "N"))
        elif escolha == "5":
            dry_input = input("Modo seguro DRY-RUN (apenas mover cursor sem clicar)? (S/N) [padrão: S]: ").strip().upper()
            testar_fluxo("login", dry_run=(dry_input != "N"))
        elif escolha == "6":
            disponiveis = listar_templates_disponiveis()
            print(f"\nTemplates encontrados ({len(disponiveis)}):")
            for t in disponiveis:
                print(f"  - {t}")
        else:
            print("Opção inválida. Tente novamente.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Testador manual de detecção e clique em imagens")
    parser.add_argument("--image", "-i", type=str, help="Nome da imagem a testar (ex: up.png, confirm.png)")
    parser.add_argument("--scan", "-s", action="store_true", help="Escaneia a tela atual e exibe ranking de todas as imagens")
    parser.add_argument("--flow", "-f", choices=["up", "down", "login"], help="Testa um fluxo completo")
    parser.add_argument("--threshold", "-t", type=float, default=0.7, help="Limiar de confiança (padrão: 0.70)")
    parser.add_argument("--timeout", type=int, default=15, help="Tempo limite de busca em segundos (padrão: 15s)")
    parser.add_argument("--click", action="store_true", help="Efetua o clique real (por padrão no CLI é dry-run se não especificado)")
    parser.add_argument("--delay", type=int, default=3, help="Segundos de contagem regressiva antes de iniciar (padrão: 3s)")
    args = parser.parse_args()

    # Se nenhum argumento de ação for passado, entra no menu interativo
    if not args.image and not args.scan and not args.flow:
        menu_interativo()
        return

    if args.scan:
        escanear_tela(threshold=args.threshold)
    elif args.flow:
        testar_fluxo(args.flow, dry_run=not args.click)
    elif args.image:
        testar_clique_imagem(
            nome_imagem=args.image,
            threshold=args.threshold,
            timeout=args.timeout,
            dry_run=not args.click,
            delay_antes=args.delay,
        )


if __name__ == "__main__":
    main()
