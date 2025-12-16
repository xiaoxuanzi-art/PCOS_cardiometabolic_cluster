setwd("D:/OneDrive/前桌面")
data_1494<-read.csv("retro_data.csv",header = T)
 library(factoextra)
##选择聚类数目
fviz_nbclust(data_1494,kmeans,method="wss")
fviz_nbclust(data_1494,kmeans,method="silhouette")
fviz_nbclust(data_1494,kmeans,method="gap_stat",iter.max=100)
set.seed(123)
kmeans_result<-kmeans(data_1494,centers =3,iter.max = 100,nstart = 100)
centroids <- kmeans_result$centers
print(centroids)
data_1494$cluster_3<-kmeans_result$cluster
write.table(data_1494, file = "D:/OneDrive/前桌面/r_result.csv", row.names = F, quote = F)
library(cluster)
set.seed(123)
silhouette_avg <- silhouette(kmeans_result$cluster, dist(data))
silhouette_avg_value <- mean(silhouette_avg[, "sil_width"])
silhouette_avg_value
library(stats)
set.seed(123)
SSE <- sum(kmeans_result$withinss)
SSE 
library(fpc)
dist_matrix <- dist(data) 
ch_score <- cluster.stats(dist_matrix, kmeans_result$cluster)
ch_score$ch
library(fpc)
set.seed(123)
clusterboot(data_1494,B = 2000, bootmethod = "boot",clustermethod = kmeansCBI,
            k =10)
library(Rtsne)
set.seed(321)
data.uni = unique(data) 
kmeans_result<-kmeans(data.uni,centers =10,iter.max = 100,nstart = 100)
tsne_out = Rtsne(data_1494[1:6],dims = 2, pca = T,max_iter = 1000,theta = 0.5,perplexity = 20,verbose = F) 
tsne_out
library(ggplot2)
tsne_result = as.data.frame(tsne_out$Y)
colnames(tsne_result) = c("tSNE1","tSNE2")
data_1494$r <- as.factor(data_1494$r)
data_1494$twostep <- as.factor(data_1494$twostep)
ggplot(tsne_result,aes(tSNE1,tSNE2,color=data_1494$r))+
  geom_point() +theme(panel.background = element_blank()) +
  theme(axis.title = element_blank(), axis.text = element_blank())+scale_color_manual(values = c("#f9f39b","#c0e2fd","#fadadd"))+
  labs(color = "Category")
library(aricode)
cluster_results<-vector("list",100)
ari_values<-numeric(0)                                  
ami_values<-numeric(0)
for(i in 1:100){
  set.seed(i)
  k_res<-kmeans(data_1494,centers=3,nstart = 1)
  cluster_results[[i]]<-k_res$cluster}
for (i in 1:99){
  for (j in (i+1):100) {
    ari<-ARI(c1=cluster_results[[i]],c2=cluster_results[[j]])
    ami<-AMI(c1=cluster_results[[i]],c2=cluster_results[[j]])
    ari_values<-c(ari_values,ari)
    ami_values<-c(ami_values,ami)}}
list(ari_mean=mean(ari_values),
     ari_sd=sd(ari_values),
     ami_mean=mean(ami_values),
     ami_sd=sd(ami_values),
     all_ari=ari_values,
     all_ami=ami_values,
     n_comparisons=length(ari_values))
### 与100次随机聚类结果比较
set.seed(123)
ari_results<-numeric(100)
for (i in 1:100) {
  random_labels<- sample(1:3,421,replace=TRUE)
  ari_results[i]<-ARI( kmeans_result$cluster,random_labels)
}
mean_ari<-mean(ari_results)
sd_ari<-sd(ari_results)
ami_results<-numeric(100)
for (i in 1:100) {
  random_labels<- sample(1:3,421,replace=TRUE)
  ami_results[i]<-AMI( kmeans_result$cluster,random_labels)
}
mean_ami<-mean(ami_results)
sd_ami<-sd(ami_results)

                                  