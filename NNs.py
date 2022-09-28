
class MobileNet3D(hk.Module):
    def __init__(self,width,modes1=8,modes2=8,modes3=8, padding=3,name="PaperNetwork"):
        
        super().__init__(name=name)
        self.modes1= modes1
        self.modes2= modes2
        self.modes3= modes3
        self.width = width
        self.padding=6

        self.w4 = hk.Conv3D(3,  1)
        
    def __call__(self,x,a):

        def ConvBlock(width,x):
            x = jax.numpy.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding),(0,0)),mode='wrap')
            x = hk.Conv3D(width,3)(x)
            x = jax.nn.gelu(x)
            print(x.shape)
            
            x = x[jax.numpy.newaxis,...]
            x = hk.DepthwiseConv3D(channel_multiplier=1,kernel_shape=3)(x)
            x = np.squeeze(x, axis=0)
            x = x[start_p:end_p,start_p:end_p,start_p:end_p,:]


            x = jax.numpy.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding),(0,0)),mode='wrap')
            x = hk.Conv3D(width,3)(x)
            x = jax.nn.gelu(x)
            x = x[start_p:end_p,start_p:end_p,start_p:end_p,:]
            
            return x

        start_p=int(self.padding)
        end_p=-start_p
        #x = x[...,jax.numpy.newaxis]
        print("Initial Shape: ",x.shape)
        x2= ConvBlock(self.width,x)
        print(x2.shape)
    
        x=x2

        for i in range(5):
            x_skip=x
            x2= ConvBlock(self.width,x2)

            print(x2.shape)
            x = x2 + x_skip


       
        #Final Convolution
        x = jax.numpy.pad(x, ( (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding),(0, 0)),mode='wrap')
        x = self.w4(x)
        x= x[start_p:end_p,start_p:end_p,start_p:end_p,:]
        print(x.shape)


        return x
        


class SpectralConv3D(hk.Module):

  def __init__(self, in_channels=1, out_channels=1, modes1=32, modes2=32, modes3=32, is_training=True, name='Layer'):

    super().__init__(name=name)
    self.name= name
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
    self.modes2 = modes2
    self.modes3 = modes3
    
    self.scale = (1 / (self.in_channels * self.out_channels))
    self.weights1= hk.get_parameter(str(self.name)+ "w1", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    self.weights2= hk.get_parameter(str(self.name)+ "w2", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    self.weights3= hk.get_parameter(str(self.name)+ "w3", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    self.weights4= hk.get_parameter(str(self.name)+ "w4", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())


  def __call__(self, pot_k):

    def compl_mul3d(input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return jnp.einsum("bixyz,ioxyz->boxyz", input, weights)
    
  
        
    self.weights1=self.weights1*self.scale
    self.weights2=self.weights2*self.scale
    self.weights3=self.weights3*self.scale
    self.weights4=self.weights4*self.scale

    batchsize=1
    
    
    #x_ft=x_ft.reshape(1,1,64,64,33)
    x_ft=pot_k
    
    _,_,dim1,dim2,dim3=x_ft.shape
    
    out_ft=jnp.zeros([batchsize, self.out_channels, dim1, dim2, dim3], dtype=float)
    out_ft=out_ft.at[:, :, :self.modes1, :self.modes2, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1))

    out_ft=out_ft.at[:, :, -self.modes1:, :self.modes2, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2))

    out_ft=out_ft.at[:, :, :self.modes1, -self.modes2:, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3))

    out_ft=out_ft.at[:, :, -self.modes1:, -self.modes2:, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4))


    return out_ft

class FNO3D(hk.Module):
    def __init__(self,width,modes1=8,modes2=8,modes3=8, padding=3,name="PaperNetwork"):
        
        super().__init__(name=name)
        self.modes1= modes1
        self.modes2= modes2
        self.modes3= modes3
        self.width = width
        self.padding=6
        self.conv0 = SpectralConv3D(2, self.width, self.modes1, self.modes2, self.modes3, name='l0')
        self.conv1 = SpectralConv3D(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l1')
        self.conv2 = SpectralConv3D(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l2')
        self.conv3 = SpectralConv3D(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l3')
        self.w0 = hk.Conv3D(self.width, 3)
        self.w1 = hk.Conv3D(self.width, 3)
        self.w2 = hk.Conv3D(self.width, 3)
        self.w3 = hk.Conv3D(self.width, 3)
        # self.w5 = hk.Conv3D(self.width, 3)
        self.w4 = hk.Conv3D(3,  1)
        
    def __call__(self,pot_k):



        start_p=int(self.padding)
        end_p=-start_p
        x=pot_k
        #print("Start",x.shape)
        _,dim1,dim2,dim3=x.shape
        x_skip=x
        
        #Fourier Space
        x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        x1 = x1[jax.numpy.newaxis,...]
        x1 = self.conv0(x1)
        x1 = np.squeeze(x1, axis=0)
        x1 = jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        x1 = jax.nn.gelu(x1)


        x = jax.numpy.pad(x, ((0,0),(self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),mode='wrap')
        #Real Space
        x=jax.numpy.transpose(x, (1,2,3,0))
        x2 = self.w0(x)
        x2 = jax.nn.gelu(x2)
        x2 = jax.numpy.transpose(x2, (3,0,1,2))
        x2= x2[:,start_p:end_p,start_p:end_p,start_p:end_p]
        print(x2.shape)

        x = x1 + x2 #+ x_skip


        x_skip=x

        #Fourier Space
        x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        x1 = x1[jax.numpy.newaxis,...]
        x1 = self.conv1(x1)
        x1=np.squeeze(x1, axis=0)
        x1=jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        x1 = jax.nn.gelu(x1)

        x = jax.numpy.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),mode='wrap')
        #Real Space
        x=jax.numpy.transpose(x, (1,2,3,0))
        x2 = self.w1(x)
        x2 = jax.nn.gelu(x2)

        x2=jax.numpy.transpose(x2, (3,0,1,2))
        x2= x2[:,start_p:end_p,start_p:end_p,start_p:end_p]

        
        x = x1 + x2 + x_skip


        x_skip=x
        #Fourier Space
        x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        x1 = x1[jax.numpy.newaxis,...]
        x1 = self.conv2(x1)
        x1=np.squeeze(x1, axis=0)
        x1=jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        x1 = jax.nn.gelu(x1)


        x = jax.numpy.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),mode='wrap')
        #Real Space
        x=jax.numpy.transpose(x, (1,2,3,0))
        x2 = self.w2(x)
        x2 = jax.nn.gelu(x2)
        x2=jax.numpy.transpose(x2, (3,0,1,2))
        x2= x2[:,start_p:end_p,start_p:end_p,start_p:end_p]

        
        x = x1 + x2 + x_skip
        
        print(x.shape)
        
        x_skip=x
        #Fourier Space
        x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        x1 = x1[jax.numpy.newaxis,...]
        x1 = self.conv3(x1)
        x1=np.squeeze(x1, axis=0)
        x1=jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        x1 = jax.nn.gelu(x1)

        x = jax.numpy.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),mode='wrap')
        #Real Space
        x=jax.numpy.transpose(x, (1,2,3,0))
        x2 = self.w3(x)
        x2 = jax.nn.gelu(x2)
        x2=jax.numpy.transpose(x2, (3,0,1,2))
        x2= x2[:,start_p:end_p,start_p:end_p,start_p:end_p]

        x = x1 + x2 + x_skip
               
        #Final Convolution
        x = jax.numpy.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)),mode='wrap')
        x=jax.numpy.transpose(x, (1,2,3,0))
        x = self.w4(x)
        #x = jax.nn.relu(x)
        x= x[start_p:end_p,start_p:end_p,start_p:end_p,:]
        print(x.shape)


        return x
