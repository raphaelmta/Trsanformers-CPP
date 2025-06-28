// Construindo Um LLM a Partir do Zero com Arquitetura Transformers em C++

// Inclui o arquivo de cabeçalho onde a classe VectorMath é definida
#include "../include/VectorOp.hpp"
#include <stdexcept> // Para std::invalid_argument

// Função que realiza a transposição de uma matriz (troca linhas por colunas)
std::vector<std::vector<double>> VectorMath::transpose(const std::vector<std::vector<double>>& matrix) {
    
    // CORREÇÃO: Verifica se a matriz está vazia para evitar erro de acesso
    if (matrix.empty() || matrix[0].empty()) {
        return {};
    }

    // Inicializa a matriz transposta com dimensões trocadas (colunas viram linhas e vice-versa)
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    
    // Itera sobre a matriz original e preenche a transposta
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Transpõe o elemento (i, j) para (j, i)
            result[j][i] = matrix[i][j]; 
        }
    }
    
    // Retorna a matriz transposta
    return result;
}

// Função que realiza a multiplicação de duas matrizes (a * b)
std::vector<std::vector<double>> VectorMath::matmul(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    
    // CORREÇÃO: Verificações de robustez
    if (a.empty() || a[0].empty() || b.empty() || b[0].empty()) {
        return {};
    }

    size_t a_cols = a[0].size();
    size_t b_rows = b.size();

    // CORREÇÃO: Valida se as dimensões são compatíveis para multiplicação
    if (a_cols != b_rows) {
        throw std::invalid_argument("As dimensões das matrizes são incompatíveis para multiplicação.");
    }

    // Inicializa a matriz resultante com o número de linhas de 'a' e o número de colunas de 'b'
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b[0].size(), 0.0));
    
    // Realiza a multiplicação de matrizes
    for (size_t i = 0; i < a.size(); ++i) {           // Itera sobre as linhas da matriz 'a'
        for (size_t j = 0; j < b[0].size(); ++j) {    // Itera sobre as colunas da matriz 'b'
            for (size_t k = 0; k < a_cols; ++k) {     // Itera sobre as colunas de 'a' (ou linhas de 'b')
                result[i][j] += a[i][k] * b[k][j];    // Calcula o produto escalar entre a linha de 'a' e a coluna de 'b'
            }
        }
    }
    
    // Retorna a matriz resultante da multiplicação
    return result;
}