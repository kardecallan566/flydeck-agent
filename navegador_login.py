import time
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


def encontrar_e_clicar_com_mss(template_path, threshold=0.9, popup=False, timeout=30):
    if cv2 is None or mss is None or pyautogui is None:
        print(f"  [Simulação] Clicar em {template_path} (cv2/mss/pyautogui ausente)")
        return

    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"Erro ao carregar imagem: {template_path}")
        return

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[1], template.shape[0]
    start_time = time.time()

    with mss.mss() as sct:
        monitor = sct.monitors[0]

        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            print(f"Confiança detectada: {max_val:.3f}")
            if max_val >= threshold:
                if popup:
                    top_right_x = monitor['left'] + max_loc[0] + 374
                    top_right_y = monitor['top'] + max_loc[1] - 130
                    print(f"Imagem encontrada! Clicando em ({top_right_x}, {top_right_y})")
                    pyautogui.moveTo(top_right_x, top_right_y, duration=0.4)
                    time.sleep(1)
                    pyautogui.click()
                    return
                center_x = monitor['left'] + max_loc[0] + w // 2
                center_y = monitor['top'] + max_loc[1] + h // 2
                print(f"Imagem encontrada! Clicando em ({center_x}, {center_y})")
                pyautogui.moveTo(center_x, center_y, duration=0.1)
                time.sleep(1)
                pyautogui.click()
                return
            else:
                if time.time() - start_time > timeout:
                    print(f"Timeout ({timeout}s) para encontrar {template_path}")
                    return
                print("Imagem não encontrada, tentando novamente...")
                time.sleep(1)


def encontrar_imagem(template_path, threshold=0.9, timeout=15):
    if cv2 is None or mss is None:
        return False

    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"Erro ao carregar imagem: {template_path}")
        return False

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    start = time.time()

    with mss.mss() as sct:
        monitor = sct.monitors[0]

        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            print(f"Confiança detectada: {max_val:.3f}")
            if max_val >= threshold:
                return True

            if time.time() - start > timeout:
                print(f"Tempo limite de {timeout}s atingido. Imagem não encontrada: {template_path}")
                return False
            time.sleep(0.5)


def encontrar_qualquer_imagem(template_paths, threshold=0.9, timeout=10, popup=False):
    if cv2 is None or mss is None or pyautogui is None:
        print(f"  [Simulação] Encontrar qualquer imagem em {template_paths}")
        return False

    templates = []
    for path in template_paths:
        template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if template is None:
            print(f"Erro ao carregar imagem: {path}")
            continue
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        templates.append((path, template_gray, template.shape[1], template.shape[0]))

    if not templates:
        print("Nenhum template válido foi carregado.")
        return False

    start = time.time()

    with mss.mss() as sct:
        monitor = sct.monitors[0]

        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            for path, template_gray, w, h in templates:
                result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                print(f"[{path}] Confiança detectada: {max_val:.3f}")
                if max_val >= threshold:
                    if popup:
                        top_right_x = monitor['left'] + max_loc[0] + 374
                        top_right_y = monitor['top'] + max_loc[1] - 130
                        print(f"Imagem encontrada! Clicando em ({top_right_x}, {top_right_y})")
                        pyautogui.moveTo(top_right_x, top_right_y, duration=0.4)
                        time.sleep(1)
                        pyautogui.click()
                        return True
                    center_x = monitor['left'] + max_loc[0] + w // 2
                    center_y = monitor['top'] + max_loc[1] + h // 2
                    print(f"Imagem encontrada! Clicando em ({center_x}, {center_y})")
                    pyautogui.moveTo(center_x, center_y, duration=0.1)
                    time.sleep(1)
                    pyautogui.click()
                    return True

            if time.time() - start > timeout:
                print("Tempo limite atingido. Nenhuma imagem encontrada.")
                return False
            time.sleep(0.5)
