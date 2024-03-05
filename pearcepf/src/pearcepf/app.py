"""
Pearce Diagram extended with Particle Flexibility
"""
import json
import pickle
import sqlite3
import sys
import re
import os
import numpy as np
import itertools
import math

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections


try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QAbstractItemView, QMainWindow, QApplication, QMenu, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QVariantAnimation, Qt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtGui import QGuiApplication

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)
working_directory = os.path.dirname(current_file_path)
# 改变当前工作目录
os.chdir(current_directory)
class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df
        self._changed = False
        self._filters = {}
        self._sortBy = []
        self._sortDirection = []

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError,):
                return None
        elif orientation == Qt.Vertical:
            try:
                return self._df.index.tolist()[section]
            except (IndexError,):
                return None

    def data(self, index, role):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            try:
                return str(self._df.iloc[index.row(), index.column()])
            except:
                pass
        elif role == Qt.CheckStateRole:
            return None

        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        dtype = self._df[col].dtype
        if dtype != object:
            value = None if value == '' else dtype.type(value)
        self._df.at[row, col] = value
        self._changed = True
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        try:
            self._df.sort_values(colname, ascending=order == Qt.AscendingOrder, inplace=True)
        except:
            pass
        try:
            self._df.reset_index(inplace=True, drop=True)
        except:
            pass
        self.layoutChanged.emit()


class CustomQTableView(QTableView):
    df = pd.DataFrame()
    def __init__(self, *args):
        super().__init__(*args)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers |
                             QAbstractItemView.DoubleClicked)
        self.setSortingEnabled(True)

    def keyPressEvent(self, event):  # Reimplement the event here
        return

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        copyAction = QAction("Copy", self)
        contextMenu.addAction(copyAction)
        copyAction.triggered.connect(self.copySelection)
        contextMenu.exec_(event.globalPos())

    def copySelection(self):
        selection = self.selectionModel().selection().indexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = '\n'.join('\t'.join(row) for row in table)
            QGuiApplication.clipboard().setText(stream)


class AppForm(QMainWindow):
    def __init__(self, parent=None, df=pd.DataFrame(),title = 'AppForm'):
        self.df = df
        self.title = title 
        self.FileName_Hint = title
        QMainWindow.__init__(self, parent)
        self.setWindowTitle(self.title)
        self.create_main_frame()

    def create_main_frame(self):        
        self.resize(400, 600) 
        self.main_frame = QWidget()
        self.table = CustomQTableView()
        model = PandasModel(self.df)
        self.table.setModel(model)  # 设置表格视图的模型
        self.save_button = QPushButton('&Save')
        self.save_button.clicked.connect(self.saveDataFile)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.table)
        self.vbox.addWidget(self.save_button)
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)

    def saveDataFile(self):

        DataFileOutput, ok2 = QFileDialog.getSaveFileName(self,
                                                          'Save Data File',
                                                          working_directory + self.FileName_Hint,
                                                          'CSV Files (*.csv);;Excel Files (*.xlsx)')  # 数据文件保存输出

        if "Label" in self.df.columns.values.tolist():
            self.df = self.df.set_index('Label')

        if (DataFileOutput != ''):

            if ('csv' in DataFileOutput):
                self.df.to_csv(DataFileOutput, sep=',', encoding='utf-8')

            elif ('xls' in DataFileOutput):
                self.df.to_excel(DataFileOutput)


class QSwitch(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setRange(0, 1)
        self.setFixedSize(60, 20)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.value() > 0.5:
            self.setValue(1)
        else:
            self.setValue(0)

class Pearce_Extended(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.init_data()
        self.generate_polygon()
        self.init_ui()

    def init_data(self):
        self.df = pd.DataFrame()        
        self.dpi = 50
        self.tag = 'VOL'
        self.setting = 'Withlines'
        self.color_setting = ''
        self.data_path=''
        self.FeO_Total = False

    def init_ui(self):
        self.setWindowTitle('Pearce-PF: Pearce Diagram extended with Particle Flexibility ')
        self.resize(1024, 600)  # 设置窗口尺寸为1024*600

        # 创建工具栏
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        self.main_frame = QWidget()
        # 创建一个空的QWidget作为间隔
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # 在工具栏中添加一个Open action
        open_action = QAction('Open Data', self)
        open_action.setShortcut('Ctrl+O')  # 设置快捷键为Ctrl+O
        open_action.triggered.connect(self.open_file)  # 连接到open_file方法
        toolbar.addAction(open_action)

        # 在工具栏中添加一个Clear action
        clear_action = QAction('Clear Data', self)
        clear_action.setShortcut('Ctrl+C') # 设置快捷键为Ctrl+C
        clear_action.triggered.connect(self.clear_data)  # 连接到clear_data方法
        toolbar.addAction(clear_action)

        # # 在工具栏中添加一个Connect action
        # connect_action = QAction('Connect', self)
        # toolbar.addAction(connect_action)

        # # 在工具栏中添加一个Load action
        # load_action = QAction('Load', self)
        # toolbar.addAction(load_action)

        toolbar.addWidget(spacer) # Add a separator

        # 在工具栏中添加一个Plot action
        plot_action = QAction('Plot Data', self)
        plot_action.setShortcut('Ctrl+P') # 设置快捷键为Ctrl+P
        plot_action.triggered.connect(self.plot_data)  # 连接到plot_data方法
        toolbar.addAction(plot_action)
        
        toolbar.addWidget(spacer) # Add a separator before the first switch


        # 在工具栏中添加一个Save action
        save_action = QAction('Save Plot', self)
        save_action.setShortcut('Ctrl+S')  # 设置快捷键为Ctrl+S
        save_action.triggered.connect(self.save_figure)  # 连接到save_figure方法
        toolbar.addAction(save_action)

        # # 在工具栏中添加一个Export action
        # export_action = QAction('Export Result', self)
        # export_action.setShortcut('Ctrl+E')  # 设置快捷键为Ctrl+E
        # export_action.triggered.connect(self.export_result)  # 连接到export_result方法
        # toolbar.addAction(export_action)


        # 创建一个表格视图
        self.table = CustomQTableView()

        # 创建一个Matplotlib画布
        self.fig = Figure((10,10), dpi=self.dpi)

        self.canvas = FigureCanvas(self.fig)

        # 设置canvas的QSizePolicy
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.canvas.setSizePolicy(sizePolicy)
        # 创建一个水平布局并添加表格视图和画布
        base_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.left_layout.addWidget(self.table)        
        self.right_layout.addWidget(self.canvas)
        base_layout.addLayout(self.left_layout)
        base_layout.addLayout(self.right_layout)

        # 创建一个QWidget，设置其布局为我们刚刚创建的布局，然后设置其为中心部件
        self.main_frame.setLayout(base_layout)
        self.setCentralWidget(self.main_frame)
        self.show()

    def resizeEvent(self, event):
        # 获取窗口的新大小
        new_width = event.size().width()
        new_height = event.size().height()

        # 设置canvas的新大小，使其始终占据窗口宽度的一半
        self.canvas.setFixedWidth(new_width / 2)

        # 调用父类的resizeEvent方法，以确保其他部件也能正确地调整大小
        super().resizeEvent(event)

    def generate_polygon(self):
        self.Polygon_dict = {}
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)

        # 获取当前文件的目录
        current_directory = os.path.dirname(current_file_path)
        # 改变当前工作目录
        os.chdir(current_directory)


    def open_file(self):
        global working_directory 
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'CSV Files (*.csv);;Excel Files (*.xls *.xlsx)')
        if file_name:
            working_directory = os.path.dirname(file_name)+'/'
            if file_name.endswith('.csv'):
                self.df = pd.read_csv(file_name)
            elif file_name.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_name)        
            
            
            model = PandasModel(self.df)
            self.table.setModel(model)  # 设置表格视图的模型

    
    def clear_data(self):
        # 清空数据
        self.df = pd.DataFrame()
        self.table.setModel(PandasModel(self.df))

        # 清空图表
        self.canvas.figure.clear()
        self.canvas.draw()


    def plot_data(self):
        if self.df.empty:
            pass
        else:
            # tag='VOL'
            # setting= 'Nolines'
            tag = self.tag
            setting = self.setting
            color_setting = self.color_setting

            # 获取当前文件的绝对路径
            current_file_path = os.path.abspath(__file__)

            # 获取当前文件的目录
            current_directory = os.path.dirname(current_file_path)
            # 改变当前工作目录
            os.chdir(current_directory)

            # 找到所有包含'PPM'或'ppm'的列名
            trace_cols = self.df.filter(regex='(PPM|ppm)', axis=1).columns

            # 创建一个新的列名映射
            new_cols = {col: re.sub(r'[_\(\)]|(PPM|ppm)', '', col, flags=re.IGNORECASE) for col in trace_cols}

            # 重命名列
            self.df.rename(columns=new_cols, inplace=True)

            # 获取Figure对象
            self.canvas.figure.clf()
            fig = self.canvas.figure
            # 生成对应个数的subplot
            ax_list = fig.subplots(2,2)

            # 现在，ax_list是一个包含4个subplot的列表
            # ax = self.canvas.figure.add_subplot(111)

            with open('Pearson_cord.json', 'r', encoding='utf-8') as file:
                cord = json.load(file)

            keys = ['coords0', 'coords1', 'coords2', 'coords3']

            for i, (ax, key) in enumerate(zip(ax_list.flatten(), keys)):
                coords = cord[key]
                for j, line in enumerate(coords["BaseLines"]):
                    start, end = line
                    linestyle = '--' if key == 'coords3' and j == len(coords["BaseLines"]) - 1 else '-'
                    # ax.plot([np.log10(start[0]), np.log10(end[0])], [np.log10(start[1]), np.log10(end[1])], color='gray', linestyle=linestyle)
                    ax.plot([start[0], end[0]], [start[1], end[1]], color='gray', linestyle=linestyle)
                for loc, label in zip(coords["LabelsLocations"], coords["Labels"]):
                    ax.text(loc[0], loc[1], label)
                ax.set_xlabel(coords["xLabel"])
                ax.set_ylabel(coords["yLabel"])
                ax.set_xscale('log')
                ax.set_yscale('log')


            self.df['Y+Nb'] = self.df['Y'] + self.df['Nb']
            self.df['Yb+Ta'] = self.df['Yb'] + self.df['Ta']

            target_x_list = ['Y+Nb','Yb+Ta','Y','Yb']
            target_y_list = ['Rb','Rb','Nb','Ta']
            
            # 遍历ax_list和target_y_list
            # for ax, target_y in zip(ax_list.flatten(), target_y_list):

            # 遍历ax_list和target_y_list
            for ax, target_x,target_y in itertools.zip_longest(ax_list.flatten(), target_x_list, target_y_list):
                # 检查target_y的值，如果是None，就跳过绘图
                if target_y is None:
                    ax.clear()
                    ax.axis('off')
                    continue

                # # 在每一个subplot上绘图
                # ax.plot(self.df['SiO2(wt%)'], self.df[target_y])

                try:

                    x = self.df[target_x]
                    y = self.df[target_y] 

                    # 如果self.df中没有'Color'列，根据'label'生成颜色
                    if 'Color' not in self.df.columns:
                        if 'Label' in self.df.columns:
                            unique_labels = self.df['Label'].unique()
                            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                            color_dict = dict(zip(unique_labels, colors))
                            self.df['Color'] = self.df['Label'].map(color_dict)
                        else:
                            self.df['Color'] = 'b'  # 默认颜色为蓝色
                    
                    # 如果self.df中没有'Marker'列，根据'label'生成符号
                    if 'Marker' not in self.df.columns:
                        if 'Label' in self.df.columns:
                            unique_labels = self.df['Label'].unique()
                            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']  # 可用的matplotlib标记
                            marker_dict = dict(zip(unique_labels, markers * len(unique_labels)))  # 如果标签数量超过标记类型，会循环使用标记
                            self.df['Marker'] = self.df['Label'].map(marker_dict)
                        else:
                            self.df['Marker'] = 'o'  # 默认符号为圆圈

                    color = self.df['Color']
                    marker = self.df['Marker']
                    alpha = self.df['Alpha'] if 'Alpha' in self.df.columns else 0.8
                    size = self.df['Size'] if 'Size' in self.df.columns else 80
                    label = self.df['Label'] 

                    # 获取当前ax对象中的所有数据点
                    for child in ax.get_children():
                        # 检查这个子对象是否是一个散点图的集合
                        if isinstance(child, collections.PathCollection):
                            # 获取当前透明度
                            current_alpha = child.get_alpha()
                            # 获取数据点的数量
                            num_points = child.get_sizes().size
                            # 根据当前透明度和数据点的数量设置新的透明度
                            if current_alpha is not None:
                                if num_points <1000:  # 如果数据点的数量大于100
                                    child.set_alpha(min(current_alpha * 2, 0.3))  # 提高透明度，但不超过1
                                elif num_points >3000:  # 如果数据点的数量小于50
                                    child.set_alpha(max(current_alpha / 2, 0.005))  # 降低透明度，但不低于0.01

                    def plot_group(group):
                        # ax.scatter(np.log10(group['x']), np.log10(group['y']), c=group['color'], alpha=group['alpha'], s=group['size'], label=group.name,edgecolors='black')
                        ax.scatter(group['x'], group['y'], c=group['color'], alpha=group['alpha'], s=group['size'], label=group.name,edgecolors='black')

                    # 创建一个新的DataFrame，包含所有需要的列
                    df = pd.DataFrame({
                        'x': x,
                        'y': y,
                        'color': color,
                        'alpha': alpha,
                        'size': size,
                        'marker': marker,
                        'label': label
                    })

                    # 按照'label'列进行分组，然后对每个组应用plot_group函数
                    df.groupby('label').apply(plot_group)

                    ax.set_xlabel(target_x)
                    ax.set_ylabel(target_y)
                    # 旋转y轴的标签
                    ax.yaxis.set_tick_params(rotation=90)
                    
                except KeyError:
                    pass

            # 获取第一个subplot的图例
            handles, labels = ax_list[0, 0].get_legend_handles_labels()

            # fig.subplots_adjust(hspace=0.5, wspace=0.5)  
            fig.tight_layout()

            fig.suptitle('Extended Pearce Diagram', fontsize=12)

            # 调整图的布局以留出足够的空间来显示图例
            fig.subplots_adjust(top=0.93)
            # 使用fig.legend()来显示图例
            fig.legend(handles, labels, loc='upper right')
            

            # Add the new canvas to the layout
            self.right_layout.addWidget(self.canvas)
            # print('fig sent to canvas')
            self.canvas.draw()
            # print('canvas drawn')

    def export_result(self):   
        if self.df.empty:
            pass
        else:
            pass                        

            df = pd.DataFrame()
            self.result_show = AppForm(df= df,title = 'Pearce Result')
            self.result_show.show()   


    def save_figure(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 
                                                   'Save Figure', 
                                                    working_directory+ 'Pearce', 'PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;PDF Files (*.pdf)')
        if file_name:
            try:
                # Set dpi to 600 for bitmap formats
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.canvas.figure.savefig(file_name, dpi=self.dpi*10)
                else:
                    self.canvas.figure.savefig(file_name)
            except Exception as e:
                # print(f"Failed to save figure: {e}")
                pass

def main():
    # Linux desktop environments use app's .desktop file to integrate the app
    # to their application menus. The .desktop file of this app will include
    # StartupWMClass key, set to app's formal name, which helps associate
    # app's windows to its menu item.
    #
    # For association to work any windows of the app must have WMCLASS
    # property set to match the value set in app's desktop file. For PySide2
    # this is set with setApplicationName().
            
    # Find the name of the module that was used to start the app
    app_module = sys.modules['__main__'].__package__
    # Retrieve the app's metadata
    metadata = importlib_metadata.metadata(app_module)

    QApplication.setApplicationName(metadata['Formal-Name'])

    app = QApplication(sys.argv)
    main_window = Pearce_Extended()
    sys.exit(app.exec())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Pearce_Extended()
    main_window.show()  # 显示主窗口
    sys.exit(app.exec())