import torch
import torch.nn.functional as F
import time as Time
    
class pinnOptimizer():
    '''
    Algorithm for processing input and do backpropagation
    '''
    
    def __init__(self, model, collocation_data, labelled_data,
                 T, L, alpha, beta2, gamma):
        self.model = model
        self.labelled_data = labelled_data
        self.collocation_data = collocation_data
        
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma 
        
        self.T = T 
        self.L = L
        
        self.loss_records = []
        self.time_records = [0]
        
    def compute_residue(self, tx: torch.Tensor) -> torch.Tensor:
        """
        Compute the residue of collocation points input
        """
        tx.requires_grad_(True) 
        uv = self.model(tx) 
        
        vu = uv[:, [1,0]] 
        
        #First derivative 
        duv_dtx = [] 
        for i in range(uv.shape[1]):
            grad = torch.autograd.grad(uv[:,i], tx, torch.ones_like(uv[:, i]), create_graph = True)[0]
            duv_dtx.append(grad)
        duv_dtx = torch.stack(duv_dtx, dim = -1)
        
        duv_dt = duv_dtx[:,0,:]
        duv_dx = duv_dtx[:,1,:]
        
        #Second derivative
        d2uv_dt2 = [] 
        for i in range(duv_dt.shape[1]):
            grad2 = torch.autograd.grad(duv_dt[:,i], tx, torch.ones_like(duv_dt[:, i]), create_graph = True)[0]
            d2uv_dt2.append(grad2)
        d2uv_dt2 = torch.stack(d2uv_dt2, dim = -1)[:,0,:]
        
        d2vu_dt2 = d2uv_dt2[:, [1,0]]
        
        scalar = torch.sum(uv**2, dim = 1, keepdim = True) 
        scalar = torch.cat([scalar, scalar], dim = 1)
        
        residue = (duv_dx/self.L + self.alpha*uv/2 - 
                   self.beta2/(2*self.T**2)*d2vu_dt2 +
                   self.gamma*scalar*vu
                  )
        return residue 
    
    
    def training_fn(self, optimizer, use_closure = False) -> float:
        self.model.train()
        
        def closure():
            start = Time.time()
            
            optimizer.zero_grad()
            uv_residue = self.compute_residue(self.collocation_data[0])
            uv_label = self.model(self.labelled_data[0])

            residue_loss = F.mse_loss(uv_residue, self.collocation_data[1], reduction = 'mean')
            data_loss = F.mse_loss(uv_label, self.labelled_data[1], reduction = 'mean')
            total_loss = residue_loss + data_loss
            
            total_loss.backward() 
            stop = Time.time()
            
            interval = stop-start
            self.loss_records.append(total_loss.item())
            self.time_records.append(self.time_records[-1]+interval)
            
            return total_loss

        
        if use_closure:
            loss = optimizer.step(closure)
        else:
            loss = closure() 
            optimizer.step()
            return loss
    
    def fit(self, adam_epochs, lbfgs_epochs):
        ADAM = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        LBFGS = torch.optim.LBFGS(self.model.parameters(), lr = 1.0,
                                       max_iter = lbfgs_epochs, history_size = 50, 
                                       line_search_fn = 'strong_wolfe')
        
        """
        Optimize the model for each epoch

        Args:
            epochs (int): number of epoch/iteration.
        """        
        total_time = 0
        print('Processing using ADAM Optimalization strategies')
        for epoch in range(adam_epochs):
            
            loss = self.training_fn(ADAM, use_closure = False)
            if (epoch+1)%10 == 0:
                print(f"Training Loss: {self.loss_records[-1]:.4e} for Epoch {epoch+1}")
            
        print('Processing using L-BFGS Optimalization strategies')
        loss = self.training_fn(LBFGS, use_closure = True)      
        print(f"Training Loss: {self.loss_records[-1]:.4e} for LBGFS Operation")