// Este script implementa um pipeline de geração de texto usando arquitetura Transfomer com as seguintes etapas:

// Tokenização: Transformando texto em tokens.
// Embedding: Convertendo tokens em vetores densos.
// Codificação Posicional: Aplicando codificações posicionais aos embeddings.
// Encoder e Decoder: Usando arquitetura Transformer para processar os inputs e gerar outputs.
// Camada Final: Aplicando uma camada final para calcular as probabilidades de tokens de saída.
// Cálculo da Perda: Usando a perda de cross-entropy para calcular o erro entre a previsão e o valor real.

// Includes
#include <iostream>                               // Biblioteca para entrada e saída de dados
#include <fstream>                                // Biblioteca para manipulação de arquivos
#include <vector>                                 // Biblioteca para vetores
#include <string>                                 // Biblioteca para strings
#include <numeric>                                // Para std::accumulate
#include <algorithm>                              // Para std::transform, std::max_element
#include <random>                                 // Para amostragem
#include <unordered_map>                          // Para o mapa de perdas
#include "./include/01RMTAEmbedding.hpp"           // Header para a classe de embeddings
#include "./include/02RMTATokenizer.hpp"           // Header para a classe de tokenização
#include "./include/03RMTAPositionalEncoding.hpp"  // Header para codificação posicional
#include "./include/04RMTALayerNorm.hpp"           // Header para normalização de camadas
#include "./include/05RMTASelfAttention.hpp"       // Header para mecanismo de self-attention
#include "./include/08RMTAEncoder.hpp"             // Header para implementação do encoder
#include "./include/10RMTADecoder.hpp"             // Header para implementação do decoder
#include "./include/FinalLayer.hpp"               // Header para a camada final de saída
#include "./include/VectorOp.hpp"                 // Header para operações de vetores

// Função para calcular a perda de cross-entropy com base nas probabilidades previstas e o token alvo
double computeCrossEntropyLoss(const std::vector<double> &predictedProbabilities, int targetTokenID)
{
    // Pegando a probabilidade prevista para o token alvo
    double predictedProbability = predictedProbabilities[targetTokenID];  

    // Epsilon para evitar log(0)
    double epsilon = 1e-9;  

    // Calculando a perda usando logaritmo
    double loss = -log(predictedProbability + epsilon);  

    return loss;
}

// Função para calcular o gradiente da perda em relação à saída da camada
std::vector<double> computeGradientOfLossWrtLayerOutput(const std::vector<double>& predictions, const std::vector<double>& trueLabels) {

    // Vetor para armazenar os gradientes
    std::vector<double> gradients(predictions.size());  

    // Calculando o gradiente para cada previsão
    for (size_t i = 0; i < predictions.size(); ++i) {

        // Diferença entre previsão e label real
        gradients[i] = predictions[i] - trueLabels[i];  
    }

    return gradients;
}

// Função principal
int main()
{
    // Inicializando o tokenizador
    Tokenizer tok;  

    // Variável para manipular o arquivo
    std::fstream file;  

    // Abrindo o arquivo de dados
    file.open("./dados/dataset.txt", std::ios::in);  

    // Vetor para armazenar as linhas do arquivo
    std::vector<std::string> text;  

    // Lendo o arquivo linha por linha
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
            {
                continue;  // Ignorando linhas vazias
            }

            text.push_back(line);  // Adicionando linha ao vetor de texto
        }
        file.close();  // Fechando o arquivo
    }
    else
    {
        // Erro se o arquivo não for encontrado
        std::cout << "Error: File not found" << std::endl;  
        return 1;  // Saída com código de erro
    }

    // Exibindo o número de linhas lidas
    std::cout << text.size() << std::endl;  

    // Vetor para armazenar as entradas
    std::vector<std::string> input_text;  
    // Vetor para armazenar as saídas esperadas
    std::vector<std::string> output_text;  
    // Vetor para armazenar os tokens das entradas
    std::vector<std::vector<int>> input_tokens;  
    // Vetor para armazenar os tokens das saídas
    std::vector<std::vector<int>> output_tokens;  


    // O loop só continua enquanto i+1 for um índice válido.
    for (size_t i = 0; i + 1 < text.size(); i += 2)
    {
        input_text.push_back(text[i]);
        output_text.push_back(text[i + 1]);
        input_tokens.push_back(tok.tokenize(text[i]));
        output_tokens.push_back(tok.tokenize(text[i + 1]));
    }

    // Definindo o token de finalização
    std::string end_token = "<end>";  
    // Pegando o ID do token de finalização
    int end_token_id = tok.tokenize(end_token)[0];  

    // Vetor para armazenar as perdas
    std::vector<std::unordered_map<int, double>> losses;  
    // Vetor para armazenar os gradientes
    std::vector<std::vector<double>> gradients;  


    int model_dim = 128;
    int vocab_size = tok.getVocabSize();
    Embedding embedding(vocab_size, model_dim);
    PositionalEncoding pe(640, model_dim);
    Encoder encoder(6, model_dim);
    Decoder decoder(6, model_dim);
    FinalLayer finalLayer(model_dim, vocab_size);

    // Loop para processar cada par de entrada e saída
    for (size_t i = 0; i < input_text.size(); i++)
    {
        // Convertendo tokens de entrada para embeddings
        std::vector<std::vector<double>> *embedded_input = embedding.tokenToEmbeddings(input_tokens[i]);  
        // Convertendo tokens de saída para embeddings
        std::vector<std::vector<double>> *embedded_output = embedding.tokenToEmbeddings(output_tokens[i]); 
        
        // Aplicando codificações posicionais aos embeddings de entrada
        std::vector<std::vector<double>> *encoded_inputs = pe.getEncodings(*embedded_input);  
        // Aplicando codificações posicionais aos embeddings de saída
        std::vector<std::vector<double>> *encoded_outputs = pe.getEncodings(*embedded_output);  

        // Passando os dados pelo encoder
        std::vector<std::vector<double>> encoder_outputs_val = encoder.forward(*encoded_inputs);  

        // Passando os dados pelo decoder
        std::vector<std::vector<double>> *decoder_outputs = decoder.forward((*encoded_inputs), encoder_outputs_val);  
        
        // Vetor para armazenar as probabilidades de saída
        std::vector<std::vector<double>> output_probabilities;  

        // Passando os outputs do decoder pela camada final para obter as probabilidades
        for (const auto &decoder_output : *decoder_outputs)
        {
            output_probabilities.push_back(finalLayer.forward(decoder_output));  
        }

        
        std::vector<int> current_output_tokens = output_tokens[i];

        if (output_probabilities.size() < current_output_tokens.size()) {
            size_t size_difference = current_output_tokens.size() - output_probabilities.size();
            // Adiciona um padding simples com distribuição uniforme para preencher
            std::vector<double> padding_probs(vocab_size, 1.0/vocab_size);
            for(size_t k = 0; k < size_difference; ++k) {
                output_probabilities.push_back(padding_probs);
            }
        } else if (output_probabilities.size() > current_output_tokens.size()) {
            size_t size_difference = output_probabilities.size() - current_output_tokens.size();
            current_output_tokens.insert(current_output_tokens.end(), size_difference, end_token_id);
        }

        // Vetor para armazenar os tokens de resposta amostrados
        std::vector<int> resp_tokens;  
        // Vetor para armazenar os tokens com maior probabilidade
        std::vector<int> resp_tokens_max;  
        // Inicializando o gerador de números aleatórios
        std::default_random_engine generator;  

        // Processando os tokens e calculando perdas
        for (size_t j = 0; j < current_output_tokens.size(); j++)
        {
            // Pegando as probabilidades de saída para o token j
            std::vector<double> output_probability = output_probabilities[j];  
            // Pegando o token alvo
            int target_token_id = current_output_tokens[j];  

            std::discrete_distribution<int> distribution(output_probability.begin(), output_probability.end());  
            int sampled_token_id = distribution(generator);  
            resp_tokens.push_back(sampled_token_id);  

            int max_token_id = std::distance(output_probability.begin(), std::max_element(output_probability.begin(), output_probability.end()));  
            resp_tokens_max.push_back(max_token_id);  

            double loss = computeCrossEntropyLoss(output_probability, target_token_id);  
            std::unordered_map<int, double> temp_loss_map;
            temp_loss_map[target_token_id] = loss;  
            losses.push_back(temp_loss_map);  
        }
        
       
        delete embedded_input;
        delete embedded_output;
        delete encoded_inputs;
        delete encoded_outputs;
        delete decoder_outputs;

        // Gerando a resposta prevista e removendo o token de finalização
        std::string response = tok.detokenize(resp_tokens);
        response = response.substr(0, response.find(end_token));

        std::cout << "Valor Previsto: \n" << response << std::endl;  

        std::string actual_response = tok.detokenize(current_output_tokens);
        actual_response = actual_response.substr(0, actual_response.find(end_token));

        std::cout << "Valor Real: \n" << actual_response << std::endl;  
        std::cout << std::endl;
    }

    // Calculando o erro total e médio
    double total_loss = 0.0;
    for (const auto &loss_map : losses)
    {
        for (const auto &loss_entry : loss_map)
        {
            total_loss += loss_entry.second;  
        }
    }

    if (!losses.empty()) {
        double average_loss = total_loss / losses.size();
        std::cout << "Erro Total: " << total_loss << std::endl;
        std::cout << "Erro Medio: " << average_loss << std::endl;
    } else {
        std::cout << "Nenhuma perda foi calculada." << std::endl;
    }

    return 0;
}