    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split as ts
     
    def Af(_x, Mode: str, Der=False):
        if Der == False:
            if Mode == "Sigm":
                return 1 / (1 + np.exp(-_x))
            elif Mode == "Relu":
                return np.where(_x > 0, _x, 0)
            elif Mode == "Tanh":
                return (2 / (1 + np.exp(_x * -2))) - 1
        elif Der == True:
            if Mode == "Sigm":
                return 1 / (1 + np.exp(-_x)) * (1 - 1 / (1 + np.exp(-_x)))
            elif Mode == "Relu":
                return np.where(_x > 0, 1, 0)
            elif Mode == "Tanh":
                return 1 - np.power((2 / (1 + np.exp(_x * -2))) - 1, 2)


    class RedeMulticamada:
            
        def __init__( self , Alpha , Entradas , Oculta , Saida , Activation ):
           
            self.Nentradas  = Entradas
            self.Noculta    = Oculta
            self.Nsaida     = Saida
            self.alpha      = Alpha
            self.Activation = Activation            
            
            self.PesosA = np.random.randn(self.Nentradas, self.Noculta) * np.sqrt(2 / self.Nentradas)
            self.PesosB = np.random.randn(self.Noculta, self.Nsaida) * np.sqrt(2 / self.Noculta)
            
            self.ViesA  = np.random.rand(self.Noculta,1)  - 0.5
            self.ViesB  = np.random.rand(self.Nsaida,1)   - 0.5
            
          
        def Alimentar( self , Inputs ):
            
            self.Inputs = Inputs
            
            self.Oculta = np.dot(self.PesosA.T,Inputs) + self.ViesA
            
            self.ZA = Af(_x = self.Oculta,Mode = self.Activation , Der = False)
    
            self.Saida = np.dot(self.PesosB.T,self.ZA) + self.ViesB
            
            self.ZB = self.Saida
    
        def Retropropagar(self, X, Y, Batch: int):

            # Variavel de normalização do batch
            N = 1 / Batch
        
            # Inicializando os acumuladores 
            AcumDerPesosA  = np.zeros_like(self.PesosA)
            AcumDerPesosB  = np.zeros_like(self.PesosB)
            AcumDerViesesA = np.zeros_like(self.ViesA)
            AcumDerViesesB = np.zeros_like(self.ViesB)
        
            for xsample, ysample in zip(X, Y):
                # Propagando para rede
                self.Alimentar(np.array([[xsample]]))
        
                # Derivada do erro quadratico
                DerEQM = ysample - self.ZB
        
                # Derivada da camada oculta em relação ao erro
                AcumDerPesosB += np.dot(self.ZA, DerEQM.T)
        
                # Derivada da camada de entrada em relação ao erro
                delta = np.dot(self.PesosB, DerEQM) * Af(self.Oculta, self.Activation, Der=True)
                AcumDerPesosA += np.dot(self.Inputs, delta.T)
        
                # Derivada das vieses em relação ao erro   
                AcumDerViesesB += DerEQM
                AcumDerViesesA += delta
        
            # Fase de ajuste dos pesos
            self.PesosB = self.PesosB + self.alpha * N * AcumDerPesosB 
            self.PesosA = self.PesosA + self.alpha * N * AcumDerPesosA
        
            # Fase de ajuste dos vieses
            self.ViesB = self.ViesB + self.alpha * N * AcumDerViesesB
            self.ViesA = self.ViesA + self.alpha * N * AcumDerViesesA

        

        
        def treinar(self, X, Y, batch_size: int, epochs: int):
            num_batches = len(X) // batch_size
        
            for epoch in range(epochs):
                
        
                for batch in range(num_batches):
                    X_batch = X[batch * batch_size:(batch + 1) * batch_size]
                    Y_batch = Y[batch * batch_size:(batch + 1) * batch_size]
                    
                    self.Retropropagar(X_batch, Y_batch,batch_size)


        def validar( self,  XTest , YTest ):
            Total = len(XTest)
            Acuracia = 0
            
            for index in range(Total):
                self.Alimentar(XTest[index])
                if abs((float(self.ZB[0][0]) - YTest[index])) < 0.05:
                    Acuracia += 1
            
            return Acuracia/Total
            
    def Sin():    
    
        np.random.seed(77)
    
        #Grafico de base 
        XSin = np.linspace(0,2*np.pi,6400)
    
        YSin = np.sin(XSin)
    
        #Zip para treinamento
        XTrain, XTest, YTrain , YTest = ts(XSin,YSin, test_size=0.3, random_state= 77)
    
        #Iniciando o Treinamento
        Episodios = 50
    
    
        Resolucao = len(XSin)
    
        Rede = RedeMulticamada(Alpha = 0.05 , Entradas = 1, Oculta = 50, Saida = 1 , Activation= "Sigm")
    
        Rede.treinar(XTrain,YTrain,1, Episodios)
    
        Acuracia = Rede.validar(XTest,YTest)
    

        Estimado = [] 
        PX = []
        PY = [] 
        for i in range(Resolucao):
            Rede.Alimentar(XSin[i])
            PX.append(XSin[i])
            PY.append(YSin[i])
            Estimado.append(float(Rede.ZB[0][0]))
        
            
        plt.style.use('dark_background')
        
        fig, comp = plt.subplots()
    
        # Plotando os pontos dos dados reais em azul
        comp.plot(PX, PY, 'b', label='Real')
        # Plotando os pontos dos dados estimados em vermelho
        comp.plot(PX, Estimado, 'y', label='Aproximado')

        # Plote alguns pontos discretos do conjunto de testes
        sample_indices = np.random.choice(len(XTest), 50, replace=False)  # Escolha 50 índices aleatórios
        comp.scatter(XTest[sample_indices], YTest[sample_indices], color='r', label='Teste', marker = 'o', s = 20 , alpha = 1)
        
        # Adicione a acurácia no gráfico
        comp.text(0.05, 0.05, f'Acurácia: {Acuracia:.3f}', transform=comp.transAxes, fontsize=16, verticalalignment='bottom')
        
        comp.legend()
        
        # Ajuste os espaços ao redor da figura
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

        # Salve a figura como uma imagem com resolução de 720p
        plt.savefig('grafico.png', dpi=600, format='png', bbox_inches='tight')


        
        plt.show()  
        
        return Rede
    if __name__ == '__main__' :
        Rede = Sin()
        