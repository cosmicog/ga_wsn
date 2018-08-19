#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "calculations.h"

extern "C"
float * runCudaPart(float *h_in, float *h_out);
double runCalcClusterHeadsAndTotalEnergy(int *h_out, float *h_in_x, float * h_in_y, int *h_in_ch, int arr_size_ch, int arr_size, int base_x, int base_y);
const char * FILE_NAME = "nodes.txt";
bool control = true;
bool first_generation = true;

void MainWindow::showSelectedChromosome(int console = 0)
{
    int index = ui->list_last->currentRow();
    if (index == -1) index = 0;
    QString * qstr = new QString("");
    char * display = new char[4096*8];
    sprintf(display, "[ Chromosome %d ] : [ ",
            index);
    char * buff = new char[32];
    for (int k = 0; k < CHRO_LEN; k++)
    {
        sprintf(buff, "%d ", LAST_POPULATION[index][k]);
        std::strcat(display, buff);
    }
    std::strcat(display, "]\n\n");

    char * chr = new char[64];

    for(int i = 0; i < POINT_COUNT; i++)
    {
        sprintf(chr, "(%d) -> %d", i, NODE_HEADS_FOR_CHRO[index][i]);
        std::strcat(display, chr);
        if(console != 0)std::cout<< chr << "\n";
        sprintf(chr, ((i % 4) != 3) ? "\t" : "\n");
        std::strcat(display, chr);
    }
    if(console != 0)std::cout<< "\n" << "----------------------" << LAST_FITNESS_VALUE[index]<<"\n";
    sprintf(chr, "\n[TOTAL ENERGY FACTOR ]: %f ", LAST_FITNESS_VALUE[index]);
    std::strcat(display, chr);

    *qstr = display;
    ui->lbl->setText(*qstr);
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //ui->statusLabel->setText(givePoints(FILE_NAME));
    giveRandomPoints();
    createRandomPopulation();
    populationsToUI();
}

void MainWindow::populationsToUI()
{
    char * chr = new char[1024];
    ui->list_start->clear();
    ui->list_last->clear();
    for (int i = 0; i < CHRO_COUNT; i++)
    {
        char * buff = new char[1024];
        sprintf(chr, "%d", START_POPULATION[i][0]);
        for (int k = 1; k < CHRO_LEN; k++)
        {
            sprintf(buff, "_%d", START_POPULATION[i][k]);
            std::strcat(chr, buff);
        }

        new QListWidgetItem(tr(chr), ui->list_start);
        sprintf(chr, "%d", LAST_POPULATION[i][0]);
        for (int k = 1; k < CHRO_LEN; k++)
        {
            sprintf(buff, "_%d", LAST_POPULATION[i][k]);
            std::strcat(chr, buff);
        }

        new QListWidgetItem(tr(chr), ui->list_last);
    }
}

bool plotSinCos()
{
    Engine * ep = engOpen (NULL);
    if (ep != NULL)
    {
        qApp->processEvents(); // For gui's stabilization

        double x[1000];
        double y[1000];
        double z[1000];

        double t = 0;
        const double dt = 0.001;
        int i,j;
        double a,b;

        mxArray *z_array = mxCreateDoubleMatrix(1000,1,mxREAL);
        mxArray *a_array = mxCreateDoubleMatrix(   1,1,mxREAL);
        mxArray *b_array = mxCreateDoubleMatrix(   1,1,mxREAL);

        double *pz = mxGetPr(z_array);
        double *pa = mxGetPr(a_array);
        double *pb = mxGetPr(b_array);

        for (i=0;i<1000;i++)
        {
            x[i] = cos(2*M_PI*t);
            y[i] = sin(2*M_PI*t);
            t+=dt;
        }

        a = 1;
        b = 0;

        for (j=0;j<100;j++)
        {
            for(i=0;i<1000;i++)
            {
                z[i] = a*x[i] + b*y[i];
                pz[i] = z[i];
            }
            pa[0] = a;
            pb[0] = b;

            engPutVariable(ep,"z",z_array);
            engPutVariable(ep,"a",a_array);
            engPutVariable(ep,"b",b_array);
            engEvalString(ep,"testPlot");

            a = a - 0.01;
            b = b + 0.01;
        }
        //engClose(ep);
    return true;
    }
    else return false;
}

bool plotChromosome(int index)
{
    Engine * ep = engOpen (NULL);
    if (ep != NULL)
    {
        qApp->processEvents();

        mxArray *x_array = mxCreateDoubleMatrix(1, POINT_COUNT, mxREAL);
        mxArray *y_array = mxCreateDoubleMatrix(1, POINT_COUNT, mxREAL);
        mxArray *c_array = mxCreateDoubleMatrix(1, CHRO_LEN, mxREAL);
        mxArray *u_array = mxCreateDoubleMatrix(1, MAX_ITERATION, mxREAL);

        double *px = mxGetPr(x_array);
        double *py = mxGetPr(y_array);
        double *pc = mxGetPr(c_array);
        double *pu = mxGetPr(u_array);

        for(int i = 0; i < POINT_COUNT; i++)
        {
            px[i] = (double) POINTS_X[i];
            py[i] = (double) POINTS_Y[i];
        }

        for(int i = 0; i < CHRO_LEN; i++)
        {
            pc[i] = LAST_POPULATION[index][i];
        }
//        std::cout << " = [";
//        for (int i = 0; i < MAX_ITERATION; i++)
//        {
//            pu[i] = ITERATION_VALUES[i];
//            std::cout << ITERATION_VALUES[i] << " ";
//        }
//        std::cout << "]"; // Burası YZ Performans grafiği için

        engPutVariable(ep,"a",x_array);
        engPutVariable(ep,"b",y_array);
        engPutVariable(ep,"c",c_array);
        engPutVariable(ep,"u",u_array);
        engEvalString(ep,"testPlot");
        //engClose(ep);
    return true;
    }
    else return false;
}

QString testCuda()
{
    QString *qstr = new QString("");
    char * display = new char[4096];
    control = false;
    const int ARRAY_SIZE = 64;
    float h_in[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = float(i);
        h_out[i] = i;
    }

    runCudaPart(h_in, h_out);

    char * asdf = new char[64];
    sprintf(display, "\n");
    for(int i=0 ; i< ARRAY_SIZE ; i++)
    {
        sprintf(asdf, "%f ", h_out[i]);
        std::strcat(display, asdf);
        sprintf(asdf, ((i % 4) != 3) ? "\t" : "\n");
        std::strcat(display, asdf);
    }

    *qstr = display;
    return *qstr;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_list_start_itemChanged(QListWidgetItem *item)
{
    int index = ui->list_start->currentRow();
    ui->statusLabel->setText(item->text() + " BE CAREFUL! Program can be crash!");
    QStringList words = item->text().split("_");
    for(int k = 0; k < CHRO_LEN; k++)
    {
        START_POPULATION[index][k] = LAST_POPULATION[index][k] = words[k].toInt();
    }
    ui->list_last->clear();
    char * chr = new char[512];
    for (int i = 0; i < CHRO_COUNT; i++)
    {

        char * buff = new char[256];
        sprintf(chr, "%d", LAST_POPULATION[i][0]);
        for (int k = 1; k < CHRO_LEN; k++)
        {
            sprintf(buff, "_%d", LAST_POPULATION[i][k]);
            std::strcat(chr, buff);
        }
        new QListWidgetItem(tr(chr), ui->list_last);
    }
}

void MainWindow::on_list_start_itemDoubleClicked(QListWidgetItem *item)
{
    item->setFlags(item->flags() | Qt::ItemIsEditable);
    ui->list_start->editItem(item);
}

void MainWindow::on_btn_calc_released()
{
    ui->btn_plot->setEnabled(false);
    ui->btn_calc->setEnabled(false);
    ui->btn_exp->setEnabled(false);
    ui->btn_calc_lvl->setEnabled(false);
    ui->list_start->setEnabled(false);
    ui->list_last->setEnabled(false);
    ui->btn_calc->setEnabled(false);
    ui->statusLabel->setText("Running, please wait...");
    ui->lbl->setText(":)");
    qApp->processEvents();
    for (int i = 0; i < MAX_ITERATION; i++)
    {
        qApp->processEvents();
        calcGeneration();
        populationsToUI();
        qApp->processEvents();
        ui->statusLabel->setText("%" + QString::number((i*100)/MAX_ITERATION) + "...");
        qApp->processEvents();
        showSelectedChromosome();
        ITERATION_VALUES[i] = LAST_FITNESS_VALUE[0];
        qApp->processEvents();
    }
    qApp->processEvents();
    populationsToUI();
    showSelectedChromosome(1);
    first_generation = false;
    ui->list_last->setEnabled(true);
    ui->statusLabel->setText("Done...");
    ui->btn_calc->setEnabled(true);
    ui->btn_exp->setEnabled(true);
    ui->btn_calc_lvl->setEnabled(true);
    ui->btn_plot->setEnabled(true);
}

void MainWindow::on_btn_exp_clicked()
{
    ui->btn_plot->setEnabled(false);
    ui->btn_exp->setEnabled(false);
    ui->btn_calc->setEnabled(false);
    ui->btn_calc_lvl->setEnabled(false);
    ui->list_start->setEnabled(false);
    ui->list_last->setEnabled(false);
    ui->btn_calc->setEnabled(false);
    ui->statusLabel->setText("Running, please wait...");
    ui->lbl->setText(":)");
    int count_level = 0;
    qApp->processEvents();
    for (int i = 0; i < MAX_ITERATION; i++)
    {
        calcGeneration();
        if (i % (MAX_ITERATION / 20) == 0)
        {
            ui->statusLabel->setText("%" + QString::number(count_level) + "...");
            count_level += 5;
            populationsToUI();
            qApp->processEvents();
            showSelectedChromosome();
            qApp->processEvents();
        }
    }
    populationsToUI();
    showSelectedChromosome(1);
    ui->list_last->setEnabled(true);
    ui->statusLabel->setText("Done...");
    first_generation = false;
    ui->btn_calc->setEnabled(true);
    ui->btn_exp->setEnabled(true);
    ui->btn_calc_lvl->setEnabled(true);
    ui->btn_plot->setEnabled(true);
}

void MainWindow::on_btn_calc_lvl_clicked()
{
    ui->btn_plot->setEnabled(false);
    ui->btn_exp->setEnabled(false);
    ui->btn_calc->setEnabled(false);
    ui->btn_calc_lvl->setText("Next\nGeneration");
    ui->list_start->setEnabled(false);
    if(first_generation)
    {
        first_generation = false;
        calcLastGenerationsFitness();
        populationsToUI();
        qApp->processEvents();
        showSelectedChromosome(1);
        qApp->processEvents();
    }
    else
    {
        calcGeneration();
        populationsToUI();
        qApp->processEvents();
        showSelectedChromosome(1);
        qApp->processEvents();
    }
    ui->btn_plot->setEnabled(true);
}

void MainWindow::on_actionReset_Program_Values_triggered()
{
    ui->btn_plot->setEnabled(false);
    ui->btn_calc->setEnabled(true);
    ui->btn_calc_lvl->setText("Calculate \nby Level");
    ui->btn_calc_lvl->setEnabled(true);
    ui->list_start->setEnabled(true);
    ui->lbl->setText("Ash nazg durbatulûk, ash nazg gimbatul!");
    ui->statusLabel->setText("Chromosomes are regenerated.");
    createRandomPopulation();
    populationsToUI();
}

void MainWindow::on_list_last_itemClicked()
{
    showSelectedChromosome(1);
}

void MainWindow::on_list_last_currentItemChanged()
{
    showSelectedChromosome(1);
}

void MainWindow::on_btn_last_graph_clicked()
{
    QPixmap pix;
    pix = QPixmap("myaa.png");
    ui->lbl->setPixmap(pix);
    ui->statusLabel->setText("Done.");
}

void MainWindow::on_btn_plot_released()
{
    ui->btn_last_graph->setEnabled(true);
    ui->btn_plot->setEnabled(false);
    ui->btn_exp->setEnabled(false);
    ui->btn_calc->setEnabled(false);
    ui->btn_calc_lvl->setEnabled(false);
    ui->statusLabel->setText("MATLAB Engine is running, please wait...");
    qApp->processEvents();
    int index = ui->list_last->currentRow();
    if (index == -1) index = 0;
    if (!plotChromosome(index))
    {
        ui->statusLabel->setText("-_-");
    }
    else
    {
        QPixmap pix;
        pix = QPixmap("myaa.png");
        ui->lbl->setPixmap(pix);
        ui->statusLabel->setText("Done.");
    }
    ui->btn_plot->setEnabled(true);
    ui->btn_exp->setEnabled(true);
    ui->btn_calc->setEnabled(true);
    ui->btn_calc_lvl->setEnabled(true);
}
