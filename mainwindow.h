#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QItemSelection>
#include <QListWidgetItem>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

  public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

  private slots:
    void populationsToUI();
    void showSelectedChromosome(int console);
    void on_btn_calc_released();
    void on_btn_plot_released();
    void on_list_start_itemChanged(QListWidgetItem *item);
    void on_list_start_itemDoubleClicked(QListWidgetItem *item);
    void on_btn_calc_lvl_clicked();
    void on_actionReset_Program_Values_triggered();
    void on_list_last_itemClicked();
    void on_list_last_currentItemChanged();
    void on_btn_exp_clicked();
    void on_btn_last_graph_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
