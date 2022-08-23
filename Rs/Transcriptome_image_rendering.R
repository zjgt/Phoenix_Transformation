library(magick)
library(ggplot2)
library(dplyr)

data <- read.csv(file=file.choose(), header=TRUE, row.names = 1, sep = ',')
n = ncol(data)
names = colnames(data)
# the first two columns are x/y coordinates of protein-coding genes
df1 = data[, 1:2]
for(i in 3:n){
  df <- cbind(df1, data[, i])
  #arrange the data according to ascending gene expression orders
  df <- arrange(df, df[, 3])
  base <-ggplot(df, aes(x=df[, 1], y=df[, 2])) + geom_point(aes(colour=df[,3]), size = 0.1) + scale_color_gradientn(colors=rainbow(5), guide=NULL) + theme_void() +xlim(0, 512) +ylim(0, 512)
  #save the transcriptome image in 3072x3072 format
  ggsave("Mock.png", plot=base, height =10.74, width = 10.74)
  obj <- magick::image_read("Mock.png")
  obj <- magick::image_crop(obj, geometry="3072x3072+72+72", gravity="NorthWest", repage=FALSE)
  magick::image_write(obj, path=sprintf("PanCanAtlas_3072/%d_%s.png",i, names[i]), format="png") 
  #save the transcriptome image in 2014x1024 format
  ggsave("Mock.png", plot=base, height =3.58, width = 3.58)
  obj <- magick::image_read("Mock.png")
  obj <- magick::image_crop(obj, geometry="1024x1024+24+24", gravity="NorthWest", repage=FALSE)
  magick::image_write(obj, path=sprintf("PanCanAtlas_1024/%d_%s.png",i, names[i]), format="png")
  #save the transcriptome image in 512x512 format
  ggsave("Mock.png", plot=base, height =1.79, width = 1.79)
  obj <- magick::image_read("Mock.png")
  obj <- magick::image_crop(obj, geometry="512x512+12+12", gravity="NorthWest", repage=FALSE)
  magick::image_write(obj, path=sprintf("PanCanAtlas_512/%d_%s.png",i, names[i]), format="png")
}