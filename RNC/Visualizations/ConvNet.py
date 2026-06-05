from common import * 
# Architecture - Basic CNN

class ConvNet(nn.Module):
    def __init__(self, input_resolution=32):
        super().__init__()

        # pooling / unpooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Network Backbone
        self.conv1 = nn.Conv2d(3, 96, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(96, 192, 3, 1, 1)
        self.bn4   = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, 1, 1)
        self.bn5   = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, 1, 1)
        self.bn6   = nn.BatchNorm2d(192)

        self.conv7 = nn.Conv2d(192, 384, 3, 1, 1)
        self.bn7   = nn.BatchNorm2d(384)
        self.conv8 = nn.Conv2d(384, 512, 3, 1, 1)
        self.bn8   = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn9   = nn.BatchNorm2d(512)

        # Dynamic channel tracking pass
        with torch.no_grad():
            # Creates a dummy tensor matching your specific dataset resolution
            dummy = torch.zeros(1, 3, input_resolution, input_resolution)

            x = F.relu(self.bn1(self.conv1(dummy)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x, _ = self.pool(x)

            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x, _ = self.pool(x)

            x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x)))
            x = F.relu(self.bn9(self.conv9(x)))

            n_channels = x.numel()


        self.fc1 = nn.Linear(n_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Output 10 classes to match the Imagenette dataset
        self.classifier = nn.Linear(in_features=128, out_features=4)

    def forward(self, x, save_deconv=False):
      x = F.relu(self.bn1(self.conv1(x)))
      if save_deconv:
          self.act1 = x.detach()
          self.size1 = x.size()

      x = F.relu(self.bn2(self.conv2(x)))
      if save_deconv:
          self.act2 = x.detach()
          self.size2 = x.size()

      x = F.relu(self.bn3(self.conv3(x)))
      if save_deconv:
          self.act3 = x.detach()
          self.size3 = x.size()

      x, idx1 = self.pool(x)

      x = F.relu(self.bn4(self.conv4(x)))
      if save_deconv:
          self.idx1 = idx1
          self.act4 = x.detach()
          self.size4 = x.size()

      x = F.relu(self.bn5(self.conv5(x)))
      if save_deconv:
          self.act5 = x.detach()
          self.size5 = x.size()

      x = F.relu(self.bn6(self.conv6(x)))
      if save_deconv:
          self.act6 = x.detach()
          self.size6 = x.size()

      x, idx2 = self.pool(x)

      x = F.relu(self.bn7(self.conv7(x)))
      if save_deconv:
          self.idx2 = idx2
          self.act7 = x.detach()
          self.size7 = x.size()

      x = F.relu(self.bn8(self.conv8(x)))
      if save_deconv:
          self.act8 = x.detach()
          self.size8 = x.size()

      x = F.relu(self.bn9(self.conv9(x)))
      if save_deconv:
          self.act9 = x.detach()
          self.size9 = x.size()

      x = torch.flatten(x,1)

      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.classifier(x)

      return x

    def keep_channel(self, act, k):
        z = torch.zeros_like(act)
        z[:, k:k+1, :, :] = act[:, k:k+1, :, :]
        return z

    def deconv(self, k, layer):

        if layer == 1:

            z = self.keep_channel(self.act1, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 2:

            z = self.keep_channel(self.act2, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 3:

            z = self.keep_channel(self.act3, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 4:

            z = self.keep_channel(self.act4, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv4.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx1,
                output_size=self.size3
            )

            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 5:

            z = self.keep_channel(self.act5, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv5.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv4.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx1,
                output_size=self.size3
            )

            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 6:

            z = self.keep_channel(self.act6, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv6.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv5.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv4.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx1,
                output_size=self.size3
            )

            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 7:

            z = self.keep_channel(self.act7, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv7.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx2,
                output_size=self.size6
            )

            z = F.conv_transpose2d(z, self.conv6.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv5.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv4.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx1,
                output_size=self.size3
            )

            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 8:

            z = self.keep_channel(self.act8, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv8.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv7.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx2,
                output_size=self.size6
            )

            z = F.conv_transpose2d(z, self.conv6.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv5.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv4.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx1,
                output_size=self.size3
            )

            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        elif layer == 9:

            z = self.keep_channel(self.act9, k)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv9.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv8.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv7.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx2,
                output_size=self.size6
            )

            z = F.conv_transpose2d(z, self.conv6.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv5.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv4.weight, padding=1)

            z = F.relu(z)
            z = self.unpool(
                z,
                self.idx1,
                output_size=self.size3
            )

            z = F.conv_transpose2d(z, self.conv3.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv2.weight, padding=1)

            z = F.relu(z)
            z = F.conv_transpose2d(z, self.conv1.weight, padding=1)

        else:
            raise ValueError("Layer is invalid")

        return z
