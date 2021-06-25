## ViT
AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
https://arxiv.org/pdf/2010.11929.pdf
这篇工作Vision Transformer基于NLP领域中大放异彩的Transformer模型来处理视觉领域的任务。作者将二维的图像数据用一个简单的方式转换为和Transformer中处理的句子序列差不多的形式， 然后使用 Transformer编码器来提取特征。

[图片上传失败...(image-dd376e-1621938122772)]

### Multi-head self-Attention 多头注意力机制
Transformer的论文叫Attention is all you need,  现在在深度学习领域中提到Attention可能大家都会想到Transformer的self-Attention自注意力，其实注意力机制刚开始是应用于循环神经网络中的，self-Attention可以看成是一个更通用的版本。Attention本来是在Encoder-Decoder框架中关乎中间的隐藏状态的这么一个函数。 而self-Attention无所谓隐藏状态，只关注输入序列中向量之间的依赖关系。Transformer给出了一个非常简洁的公式 。
![三个矩阵：Q: Query 查询，K：Key 键， V：Value 值。 d_k 是key的dimension维度进行一个缩放用来稳定训练。](https://upload-images.jianshu.io/upload_images/25769723-223b0d05707134b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

看到softmax就知道是在求概率，V代表的是数值，QK代表一个查字典的操作。但是这样还是很抽象，要理解的话得把矩阵拆成向量才行。这里推荐一篇可视化Transformer的博客。https://jalammar.github.io/illustrated-transformer/

![这里的q,k,v就是代表向量了，对于输入序列中的每个token，它对应的q会去查询所有其他token的key，得到一个可以理解为关联程度的分数，转化为概率再对每个token的Value根据关联程度进行一个权重。 最终结果和所有token的value都有关系，但是注意力会集中在关联程度大的value上面。](https://upload-images.jianshu.io/upload_images/25769723-29b844e8da02f5a9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我的理解就是把原向量进行三次编码，然后在计算attention结果的时候，一个编码只和自己有关，代表该token的特征，另外两个用来和序列中其他向量的编码进行匹配，得到当前向量与其他向量之间的关联程度。

卷积在视觉中占主流的原因很重要的原因是局部感受野，另外卷积的形式一坨一坨的很契合对图片数据的处理。但是，卷积的感受野是受限的，要多层抽象才能得到一个比较大的感受野。而自注意力我觉得可以理解为在输入的全局中有选择的进行权重。这个过程进行多次，就是多头自注意力机制。

### 把图片当作单词处理
最终的编码就长成这个样子：
![ 一张图片的embedding ](https://upload-images.jianshu.io/upload_images/25769723-c3d5d7158694bd8b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
对应：
![Embedding](https://upload-images.jianshu.io/upload_images/25769723-d9ce48fda59ad667.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1. 图片转化为序列数据 $ \mathbf {x}_p^i$E
将图片拆分为多个patch，每个压扁的 $ \mathbf {x}_p^i$ 通过线性变换E， 得到一个固定长度的特征向量，参考NLP中的习惯称为token。这个token的长度D文中使用了768，1024，1280对应三个尺寸的模型ViT-Base，Large以及Huge。

2. class token $ \mathbf {x}_{\mathbf {class}}$
另外每个序列的开头还会加上一个class token，最终用来分类的是class token 对应的特征向量。这个token的参数是可学习的，会和序列中所有其他patch所生成的token一样，正常进行查询匹配的注意力操作，我的理解是它起到了一个类似总结的作用。代码中可以看到，最终通过MLP的要么是只取class token的结果，或者也可以使用对所有token在每个位置取平均值的方法。但是论文好像没有解释取平均值会怎么样，有了解的同学欢迎补充。

```
#https://github.com/lucidrains/vit-pytorch/blob/4f3dbd003f004569a916f964caaaf7b9a0a28017/vit_pytorch/vit.py
 def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # （只使用 class token）

        x = self.to_latent(x)
        return self.mlp_head(x)
```

3. 位置编码 $E_{pos}$
虽然注意力可以捕捉到token和token之间的依赖关系，但是token的位置信息却无处可寻。也就是说，无论这些patch如何排序，得到的结果都是一样的。NLP领域中有非常多的解决方案，ViT使用的是可学习的位置编码，和class token 与 patch的线性变换相加得到最终编码。也许也可以用拼接，不过原Transformer中没有提到，另外Transformer中使用的是固定的编码。 总之，就是无论哪个序列，让序列中同一位置的token附带上一模一样的信息就可以了。ViT附录D3中有不同位置编码方式的对比实验结果，如果没有考虑位置信息，那么结果很差，而使用不同位置编码的结果其实差距不大。
![不同编码的对比试验，编码方法可以去参看原文](https://upload-images.jianshu.io/upload_images/25769723-d3cc9b100bc46e78.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




既然已经通过上面的处理把图片的输入转化为Tranformer处理单词序列的形式了，那么接下来直接通过多头注意力机制多次处理，最终得到的结果是和图片中每个patch都相关的特征。就相当于替代卷积层完成了特征提取得到 z_l。

![MSA: 多头注意力机制， MLP多层感知机](https://upload-images.jianshu.io/upload_images/25769723-8b5c17797f09968c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 实验

不用卷积运算，训练需要的计算资源要少很多。
另外作者也尝试了自监督的方法，ViT应该是也表现不错。
ViT 如果用大量数据集进行预训练，那么效果会很好。
ViT 模型更大对比同量级state-of-the-art表现更好。
![训练得到的当前patch的位置编码，和其他所有位置编码的余弦相似度，也就是做点积再除以模的积](https://upload-images.jianshu.io/upload_images/25769723-a98878c9f3e749d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## swinTransformer
https://arxiv.org/pdf/2103.14030.pdf
![先看之前的ViT，特征图始终是比较低的像素，而Swin Transformer刚开始把原图分成比较多的小窗口，然后在下一个阶段把邻近的小窗口融合成大窗口。虽然patch的数量始终保持4*4，但是patch的分辨率是从高到底变化的。 ](https://upload-images.jianshu.io/upload_images/25769723-86e6e9f81d7ef955.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![swin Transformer结构](https://upload-images.jianshu.io/upload_images/25769723-bf2159bdae231d90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![另外，某一层的窗口分界处是另一层一个窗口的中心。这个设计很好理解，对比实验证明也是有效的。](https://upload-images.jianshu.io/upload_images/25769723-50d68b4a3513ce0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![其中W-MSA和 SW-MSA代表对不同分区的patch使用多头注意力机制 ](https://upload-images.jianshu.io/upload_images/25769723-691441c206e18173.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## position bias in self-attention head 

不同于ViT中在输入序列中加上一个绝对的位置编码，swinTransformer使用的是相对位置偏置，加在attention内部的查询操作里。论文做了实验，如果同时使用两种方法，表现会反而下降。

![在原自注意力机制的基础上，加了一个偏置B，这个表示的是patch的相对位置，如果在此基础上再叠加一个绝对位置偏置，表现会反而下降](https://upload-images.jianshu.io/upload_images/25769723-b0937f74d3c597d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





