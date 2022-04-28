# AM-Backend
Para utilizar o projeto em mac e linux acesse a branch:
```
MacLinuxProject
```

# Notebook.ipynb
Este notebook é uma implementações da arquitetura encontrada no artigo [DeXpression: Deep Convolutional Neural Network for Expression Recognition](https://paperswithcode.com/paper/dexpression-deep-convolutional-neural-network), juntamente com o dataset da [fer13](https://www.kaggle.com/datasets/gauravsharma99/fer13-cleaned-dataset). A atual acuracia do modelo gerado a partir deste notebook é de 64.8%, utilizando somente 5 categorias de emoções, das 11 iniciais propostas pelo artigo em questão. Acreditamos que a falta de acuracia do modelo é devido ao tamanho improprio das imagens do dataset que foi utilizado no treinamento, juntamente com a quantidade de classes. 

A utilização do dataset [fer13](https://www.kaggle.com/datasets/gauravsharma99/fer13-cleaned-dataset) na arquitetura Dexpression, só foi possível devido algumas alterações no arquitetura original. 

## Arquitetura DeXpression

![](./imgs/arq1.png)

A arquitetura de Rede Neural Convolucional profunda proposta (representada na figura acima) consiste em quatro partes. A primeira parte pré-processa automaticamente os dados. Esse começa com a Convolução 1, que aplica 64 filtros. A próxima camada é o Pooling 1, que reduz a amostragem as imagens e então elas são normalizadas pelo LRN 1. O próximos passos são os dois FeatEx (Extração Paralela de Recursos Block), destacados na Figura 4. São os blocos núcleo da arquitetura proposta e descrito mais tarde em
esta seção. As características extraídas por esses blocos são
encaminhados para uma camada totalmente conectada, que os utiliza
classificar a entrada nas diferentes emoções.

![](./imgs/arq2.png)




