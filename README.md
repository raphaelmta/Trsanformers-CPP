# bumblebee — Transformer em C++ puro 🤖🐝

![Linguagem](https://img.shields.io/badge/Linguagem-C++20-blue.svg)
![Licença](https://img.shields.io/badge/Licença-MIT-green.svg)

> Projeto de estudo e desenvolvimento pessoal: **transformando** café em código e C++ em IA. Uma arquitetura Transformer completa (LLM) implementada do zero — sem “carona” em frameworks de machine learning! ☕🚀

---

## Sobre o Projeto

Este repositório documenta o desenvolvimento de um Modelo de Linguagem Grande (LLM) baseado na arquitetura **Transformer**, feito 100% em C++ padrão (C++20).  
O objetivo é puramente didático e experimental: estudar e entender a fundo o funcionamento interno dos Transformers modernos — como GPT ou BERT — ao construir cada componente essencial da arquitetura “na unha”, em classes C++ independentes.

> **Por que “bumblebee”?**  
> Além do trocadilho com os robôs Transformers 🦾, a ideia é mostrar que, assim como o inseto que não deveria voar mas voa, um estudante focado pode fazer IA voar só com C++ puro — sem precisar de “megatron” de frameworks! 🐝

---

## O que está implementado

- **Tokenização personalizada** (Tokenizer) — porque dividir para conquistar nunca sai de moda!
- **Embeddings** e codificação posicional — o GPS dos tokens, para ninguém se perder na sequência.
- **LayerNorm** — alinhando as energias da rede, zen total.
- **Self-Attention** (atenção própria) — aqui cada token é narcisista por natureza.
- **Feed-Forward Network** — porque, às vezes, é preciso ir direto ao ponto.
- **Encoder e Decoder** (com múltiplas camadas, compondo o modelo completo) — igual cebola, cada camada faz você chorar de alegria!
- **Cálculo da perda** (cross-entropy) — a dor faz parte do aprendizado, literalmente.
- **Pipeline funcional**: o executável principal (`bumblebee.cpp`) orquestra os módulos e mostra como conectar tudo sem transformar em bagunça.

Cada componente da arquitetura virou uma **classe C++ separada** — mais modular que braço de Transformer!

---

## Estrutura do Projeto

```text
TRANSFORMERSCPP/
├── dados/
│   ├── dataset.txt
├── include/
│   ├── 01RMTAEmbedding.hpp
│   ├── 02RMTATokenizer.hpp
│   ├── 03RMTAPositionalEncoding.hpp
│   ├── 04RMTALayerNorm.hpp
│   ├── 05RMTASelfAttention.hpp
│   ├── 06RMTAFeedForwardNetwork.hpp
│   ├── 07RMTAEncoderLayer.hpp
│   ├── 08RMTAEncoder.hpp
│   ├── 09RMTADecoderLayer.hpp
│   ├── 10RMTADecoder.hpp
│   ├── FinalLayer.hpp
│   ├── HelpFunc.hpp
│   └── VectorOp.hpp
├── src/
│   ├── 01RMTAEmbedding.cpp
│   ├── 02RMTATokenizer.cpp
│   ├── 03RMTAPositionalEncoding.cpp
│   ├── 04RMTALayerNorm.cpp
│   ├── 05RMTASelfAttention.cpp
│   ├── 06RMTAFeedForwardNetwork.cpp
│   ├── 07RMTAEncoderLayer.cpp
│   ├── 08RMTAEncoder.cpp
│   ├── 09RMTADecoderLayer.cpp
│   ├── 10RMTADecoder.cpp
│   ├── FinalLayer.cpp
│   └── VectorOp.cpp
├── bumblebee.cpp
├── bumblebee
├── LEIAME.txt
└── README.md
```


O programa lê `dados/dataset.txt`, processa os dados e imprime os resultados do modelo no console. Se der erro, não se desespere: é só dar um reboot… igual no filme!

---

## Contribuindo 🚗💨

Contribuições e sugestões são sempre bem-vindas!  
Para colaborar e entrar na “equipe dos Autobots”:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'feat: Adiciona NovaFuncionalidade'`)
4. Dê push na branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request




---

> “A melhor forma de entender um Transformer é construir um. A segunda melhor forma é… bom, pelo menos tente rodar esse repositório!”  
> — O Bumblebee (provavelmente)

Projeto mantido por Raphael Martins.  
GitHub: [@raphaelmta](https://github.com/raphaelmta)  
LinkedIn: [https://www.linkedin.com/in/raphaelmta/](https://www.linkedin.com/in/raphaelmta/)