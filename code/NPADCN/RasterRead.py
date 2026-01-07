#!C:/Python27/ArcGIS10.1/python.exe -u
# coding=UTF-8

"""
Users who redistribute the program, with or without changes, must pass along the freedom to further copy and change it. 
In addition, the orignal authors should be refered.
This is a program for reading or writing an geo-rigistered image file, programmed by Dr. Song, Xianfeng, Contact: ucas@ucas.ac.cn.
"""

from osgeo import gdal
import numpy as np


class IMAGE:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件

        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount   # 波段数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

        del dataset 

        return im_proj,im_geotrans,im_data

    # 写GeoTiff文件

    def write_img(self, filename, im_proj, im_geotrans, im_data, im_type="GTIFF"):
        # gdal数据类型
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName(im_type)
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset


# Test

if __name__ == "__main__":
    test = IMAGE()

    # 拷贝图像
    filename = "fdem.tif"
    im_proj,im_geotrans,im_data = test.read_img(filename)
    filename = "fdem_copy.tif"
    test.write_img(filename,im_proj,im_geotrans,im_data)

    # 显示数据
    print(im_data.shape)
    print(im_geotrans)

