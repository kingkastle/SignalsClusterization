########################################################################
############## Extracción de datos y Clusterizacion   ################
############### Análisis Cluster:     Kmeans         #################
########################################################################


# Cargamos las librerías necesarias y nos conectamos a la API
library(dplyr) # para conectarnos a la API de RedShift y hacer consultas internas para no hacer cálculos en local salvo necesidad.
library(data.table) # para trabajar los dataframes de forma eficiente
library("RJDBC")
options(java.parameters =  "-Xmx8192m") # para incrementar la memoria de Java


# https://blogs.aws.amazon.com/bigdata/post/Tx1G8828SPGX3PK/Connecting-R-with-Amazon-Redshift
# download Amazon Redshift JDBC driver: https://blogs.aws.amazon.com/bigdata/post/Tx1G8828SPGX3PK/Connecting-R-with-Amazon-Redshift
# download.file('http://s3.amazonaws.com/redshift-downloads/drivers/RedshiftJDBC41-1.1.9.1009.jar','RedshiftJDBC41-1.1.9.1009.jar')
# IMPORTANTE: Comprobar que la instalación de rJava es correcta para el usuario de la sesion

# https://blogs.aws.amazon.com/bigdata/post/Tx1G8828SPGX3PK/Connecting-R-with-Amazon-Redshift
driver <- JDBC("com.amazon.redshift.jdbc41.Driver", "RedshiftJDBC41-1.1.9.1009.jar", identifier.quote="`")
url <- "jdbc:redshift://XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
conn <- dbConnect(driver, url)


# Para obtener los datos de consumo para los clientes, hacemos la siguiente query:
query <- ""
TEMP <- dbGetQuery(conn, query)

# Visualizamos un summary de los resultados:
summary(TEMP)

# Normalizamos por filas (recuerda que los CUPS están distribuidos por filas) y transponemos
# Ejecuta en la consola ?apply para entender lo que hace la función
# Eliminamos la columna de CUPS del dataframe
TEMP_NORM <- t(apply(subset(TEMP, 1, function(x)(x-min(x))/(max(x)-min(x))))
TEMP_NORM <- replace(TEMP_NORM, is.na(TEMP_NORM), 0)

# Visualizamos los valores normalizados:
summary(TEMP_NORM)

# Los consumos normalizados son los que vamos a pasar al algoritmo para generar los clusters
# La variable TEMP_NORM, que es un dataframe, lo vamos a transformar en este objeto que es mas eficiente a nivel de memoria:
clusterData <- as.data.table(TEMP_NORM)
cat("\14") # Limpiar la consola
summary(clusterData)


# Vamos a probar la clusterización CLARA para poder realizarlo de manera eficiente
# Ejecuta en la consola ?clara para ver la documentacion del paquete
library(cluster)
# ks define el numero de clusters a probar:
modelresults <- clara(x = clusterData, k = 6)
# plot(modelresults)
#Ahora se elije cuál de todos es el idóneo y se calcula. En este caso: x grupos, distancia Y.     
ClusterData <- data.frame(modelresults$medoids)
ClusterData$ID <-1:nrow(ClusterData)

library(tidyr)
library(dplyr)
library(ggplot2)

tmp_long <- gather(ClusterData,hour,value,hour_0:hour_23)
tmp_long$hour <- as.character(tmp_long$hour)
tmp_long$hour[tmp_long$hour == "hour_0"] <- 0
tmp_long$hour[tmp_long$hour == "hour_1"] <- 1
tmp_long$hour[tmp_long$hour == "hour_2"] <- 2
tmp_long$hour[tmp_long$hour == "hour_3"] <- 3
tmp_long$hour[tmp_long$hour == "hour_4"] <- 4
tmp_long$hour[tmp_long$hour == "hour_5"] <- 5
tmp_long$hour[tmp_long$hour == "hour_6"] <- 6
tmp_long$hour[tmp_long$hour == "hour_7"] <- 7
tmp_long$hour[tmp_long$hour == "hour_8"] <- 8
tmp_long$hour[tmp_long$hour == "hour_9"] <- 9
tmp_long$hour[tmp_long$hour == "hour_10"] <- 10
tmp_long$hour[tmp_long$hour == "hour_11"] <- 11
tmp_long$hour[tmp_long$hour == "hour_12"] <- 12
tmp_long$hour[tmp_long$hour == "hour_13"] <- 13
tmp_long$hour[tmp_long$hour == "hour_14"] <- 14
tmp_long$hour[tmp_long$hour == "hour_15"] <- 15
tmp_long$hour[tmp_long$hour == "hour_16"] <- 16
tmp_long$hour[tmp_long$hour == "hour_17"] <- 17
tmp_long$hour[tmp_long$hour == "hour_18"] <- 18
tmp_long$hour[tmp_long$hour == "hour_19"] <- 19
tmp_long$hour[tmp_long$hour == "hour_20"] <- 20
tmp_long$hour[tmp_long$hour == "hour_21"] <- 21
tmp_long$hour[tmp_long$hour == "hour_22"] <- 22
tmp_long$hour[tmp_long$hour == "hour_23"] <- 23

tmp_long$hour=as.POSIXct(tmp_long$hour, format="%H")

png("plot1.png", height = 1080,width = 1920)
ggplot(data=tmp_long,aes(x=hour,y=value,group=ID,colour=ID))+geom_line()+ 
  scale_colour_gradientn(colours=rainbow(6)) + 
  ylab("Consumo Normalizado") + 
  xlab("Hora")
dev.off()
