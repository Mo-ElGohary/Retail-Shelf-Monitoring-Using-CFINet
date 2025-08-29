import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np

class CFINet(nn.Module):
    """
    CFINet: Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning
    Based on the ICCV'23 paper implementation
    """
    
    def __init__(self, num_classes=1, pretrained=False):
        super(CFINet, self).__init__()
        self.num_classes = num_classes
        
        # Backbone network (ResNet-50) - set pretrained=False to avoid download issues
        backbone = resnet50(pretrained=False)
        
        # Extract different layers for FPN
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels  
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels
        
        # Feature Pyramid Network (FPN)
        self.fpn = FPN([256, 512, 1024, 2048], 256)
        
        # Coarse Proposal Generator
        self.coarse_proposal_generator = CoarseProposalGenerator(256)
        
        # Fine Proposal Generator
        self.fine_proposal_generator = FineProposalGenerator(256)
        
        # Imitation Learning Module
        self.imitation_module = ImitationModule(256, num_classes)
        
        # Classification and Regression Heads
        self.cls_head = ClassificationHead(256, num_classes)
        self.reg_head = RegressionHead(256, 4)
        
    def forward(self, x):
        # Extract features from backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Get features from different layers
        c1 = self.layer1(x)      # 256 channels
        c2 = self.layer2(c1)     # 512 channels
        c3 = self.layer3(c2)     # 1024 channels
        c4 = self.layer4(c3)     # 2048 channels
        
        features = [c1, c2, c3, c4]
        
        # FPN feature extraction
        fpn_features = self.fpn(features)
        
        # Use the highest resolution feature map for proposals
        main_feature = fpn_features[0]  # Use the first (highest resolution) feature map
        
        # Coarse proposal generation
        coarse_proposals = self.coarse_proposal_generator(main_feature)
        
        # Fine proposal generation with imitation learning
        fine_proposals = self.fine_proposal_generator(main_feature, coarse_proposals)
        
        # Apply imitation learning
        refined_proposals = self.imitation_module(main_feature, fine_proposals)
        
        # Classification and regression
        cls_scores = self.cls_head(refined_proposals)
        reg_deltas = self.reg_head(refined_proposals)
        
        return cls_scores, reg_deltas, refined_proposals

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel in in_channels:
            lateral_conv = nn.Conv2d(in_channel, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, features):
        laterals = [lateral_conv(feature) for feature, lateral_conv in zip(features, self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample the higher level feature
            upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest')
            laterals[i] = laterals[i] + upsampled
        
        # FPN output
        outs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]
        return outs

class CoarseProposalGenerator(nn.Module):
    """Coarse proposal generation module"""
    
    def __init__(self, in_channels):
        super(CoarseProposalGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, 9, 1)  # 9 anchors per position
        
    def forward(self, features):
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        anchors = self.conv3(x)
        return anchors

class FineProposalGenerator(nn.Module):
    """Fine proposal generation module with attention mechanism"""
    
    def __init__(self, in_channels):
        super(FineProposalGenerator, self).__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, 9, 1)
        
    def forward(self, features, coarse_proposals):
        # Apply attention mechanism
        b, c, h, w = features.shape
        features_flat = features.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        
        attended_features, _ = self.attention(features_flat, features_flat, features_flat)
        attended_features = attended_features.permute(0, 2, 1).view(b, c, h, w)
        
        # Generate fine proposals
        x = F.relu(self.conv1(attended_features))
        x = F.relu(self.conv2(x))
        fine_proposals = self.conv3(x)
        
        return fine_proposals

class ImitationModule(nn.Module):
    """Imitation learning module for proposal refinement"""
    
    def __init__(self, in_channels, num_classes):
        super(ImitationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, num_classes + 4, 1)  # cls + reg
        
    def forward(self, features, proposals):
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        refined = self.conv3(x)
        return refined

class ClassificationHead(nn.Module):
    """Classification head for object detection"""
    
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class RegressionHead(nn.Module):
    """Regression head for bounding box regression"""
    
    def __init__(self, in_channels, num_bbox_reg):
        super(RegressionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, num_bbox_reg, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x) 