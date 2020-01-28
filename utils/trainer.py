import torch
from torch.autograd import Variable
import torch.nn.functional as F
# from utils.display_utils import image_gray

'''
save_best and save_last are paths
'''
def train_loop(train_loader, val_loader, model, optimizer, scheduler, 
               criterion,save_best, save_last, epochs, test_image=None,test_label=None ):    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    d_mean_train_losses = []
    g_mean_train_losses = []
    mean_val_losses = []

    mean_train_acc = []
    mean_val_acc = []
    minDLoss = 99999
    minGLoss = 99999
    maxValacc = -99999
    for epoch in range(2000):
        print('EPOCH: ',epoch+1)
        train_acc = []
        val_acc = []
        
        d_running_loss = 0.0
        g_running_loss = 0.0
        
        discriminator.train()
        generator.train()
        
        
        
        count = 0
        for images, labels in train_loader:    
            
    #         scheduler.step()
            
            images = Variable(images.cuda())
            labels1 = Variable(torch.ones(images.shape[0]).cuda())        
            
            #train Discriminator with real data
            discriminator_out = discriminator(images)         
            discriminator_loss = criterion(discriminator_out, labels1)
            
            #train Discriminator with fake data        
            noise = torch.from_numpy(np.random.normal(0,1, (labels.shape[0],100))).float().cuda()
            fake_label = Variable(torch.zeros(images.shape[0]).cuda())
            gen_img = generator(noise)       
            
            
            d_fake_results = discriminator(gen_img)
            d_fake_loss = criterion(d_fake_results, fake_label)
            
        
            discriminator_losses = discriminator_loss + d_fake_loss
            d_running_loss += discriminator_losses.item()
            
            discr_opt.zero_grad()  
            discriminator_losses.backward()
            discr_opt.step()        
            
            #########################################
            # train generator
            #########################################
            
            noise = torch.from_numpy(np.random.normal(0,1, (labels.shape[0],100))).float().cuda()
            g_result = generator(noise)
            d_result = discriminator(g_result)
            g_loss = criterion(d_result, labels1)
            g_running_loss += g_loss.item()
            
            discriminator.zero_grad()
            generator.zero_grad()
            g_loss.backward()
            gen_opt.step()
            
            count +=1
        
        d_ave = d_running_loss/count
        g_ave = g_running_loss/count
        print('DISCRIMINATOR Training loss:...', d_ave )
        d_mean_train_losses.append(d_ave)
        
        print('GENERATOR Training loss:...', g_ave)
        print('')
        g_mean_train_losses.append(g_ave)
        
        if d_ave < minDLoss:
            torch.save(discriminator.state_dict(), 'exp/dc_gans1/best_discr.pth')
            print('Best DLoss : ', d_ave, '....OLD : ', minDLoss)
            minDLoss = d_ave
        
            
        if g_ave < minGLoss:
            torch.save(generator.state_dict(), 'exp/dc_gans1/best_gen.pth')
            print('Best GLoss : ', g_ave, '....OLD : ', minGLoss)
            minGLoss = g_ave
        
        torch.save(generator.state_dict(), 'exp/dc_gans1/last_gen.pth' )
        
        #generate an image and display
        generator.eval()
        noise = noise = torch.from_numpy(np.random.normal(0,1, (1,100))).float().cuda()
        result = generator(noise)
    #     result.shape
        plt.figure()
        plt.subplots(figsize=(3,3))
        plt.imshow(result[0].squeeze().data.cpu().numpy(), cmap='gray')
        plt.show()

        
            

        
        print('')  