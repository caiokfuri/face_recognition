# Face Expression & Gesture Reaction

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google%20AI-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

Aplicação em Python que utiliza visão computacional em tempo real para detectar expressões faciais e gestos manuais via webcam, exibindo memes e reações correspondentes em um painel interativo lado a lado.

Compatível com **Windows**, **macOS** e **Linux**.

---

## 📸 Expressões e Imagens Mapeadas

| Expressão / Gesto | Imagem / Meme | Como Acionar |
| :--- | :--- | :--- |
| **Surpreso** | `images/surprised.jpg` | Abra a boca e arregale os olhos / levante as sobrancelhas 😮 |
| **Bocejo** | `images/yawn.jpg` | Abra bem a boca (simulação de bocejo) 🥱 |
| **Confuso / Dúvida** | `images/confused.jpg` | Incline a cabeça para o lado ou levante uma sobrancelha 🤨 |
| **Like / Positivo** | `images/like.jpg` | Faça sinal de joinha com a mão 👍 ou dê um sorriso 😄 |
| **Nerd** | `images/nerd.jpg` | Aponte o dedo indicador para cima ☝️ ("na verdade...") 🤓 |

---

## 🚀 Como Instalar e Rodar

### 1. Clonar o Repositório
```bash
git clone https://github.com/caiokfuri/face_recognition.git
cd face_recognition
```

---

### 2. Criar e Ativar o Ambiente Virtual

#### 🪟 No Windows:
```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
venv\Scripts\activate
```
*(Se o PowerShell bloquear a execução de scripts, execute `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` antes).*

#### 🍎 No macOS ou 🐧 Linux:
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate
```

---

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

---

### 4. Executar a Aplicação
```bash
python main.py
```

> [!NOTE]
> **Download Automático**: Na primeira execução, o script baixará automaticamente os modelos de IA do MediaPipe (`face_landmarker.task` e `gesture_recognizer.task`) com uma barra de progresso no terminal. Não é necessário baixar nada manualmente.

---

## ⌨️ Controles e Atalhos de Teclado

Durante a execução da janela da câmera:

- **`C`**: **Calibrar Postura Neutra**. Olhe normalmente para a câmera e pressione `C`. Ele zera a inclinação e assimetria natural do seu rosto, evitando falsos positivos.
- **`D`**: **Métricas em Tempo Real (Debug)**. Liga/desliga o painel com barras ao vivo mostrando abertura da boca, sorriso, sobrancelhas e ângulo da cabeça.
- **`M`**: **Malha Facial**. Ativa/desativa os pontos anatômicos da malha do rosto.
- **`Q`** ou **`ESC`**: Encerra a aplicação.

---

## 📁 Estrutura do Projeto

```
face_recognition/
├── src/
│   ├── config.py           # Configurações globais, resoluções e limiares calibráveis
│   ├── detectors.py        # Inicialização e inferência dos modelos MediaPipe
│   ├── classifier.py       # Algoritmo de pontuação e classificação de expressões
│   ├── ui.py               # HUD, barras de métricas, painel de reação e layout
│   └── utils.py            # Download de modelos com progresso, resize e geometria
├── images/                 # Memes / imagens de reações
├── models/                 # Modelos do MediaPipe (.task baixados automaticamente)
├── main.py                 # Ponto de entrada da aplicação
├── requirements.txt        # Dependências do projeto
├── LICENSE                 # Licença MIT
└── README.md               # Documentação do projeto
```

---

## 💡 Solução de Problemas Comuns

- **A webcam não abre**:
  - **No Windows**: Feche outros programas que usam a câmera (Teams, Zoom, Meet, etc.) e verifique em *Configurações > Privacidade > Câmera*.
  - **No macOS**: Certifique-se de que o Terminal ou VS Code tem permissão em *Ajustes do Sistema > Privacidade e Segurança > Câmera*.
  - **No Linux**: Certifique-se de que seu usuário pertence ao grupo `video` (`sudo usermod -aG video $USER`).
- **Expressão "confuso" acionando sozinha**:
  - Basta olhar normalmente para a câmera e pressionar a tecla **`C`** para calibrar sua postura neutra.

---

## 📄 Licença

Este projeto é distribuído sob a licença [MIT](LICENSE).
