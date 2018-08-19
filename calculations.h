#ifndef CALCULATIONS_H
#define CALCULATIONS_H
#include "QString"
#include "QFile"
#include "QTextStream"
#include "QStringList"
#include "QStringListModel"
#include "QMessageBox"
#include "QSizeGrip"
#include "engine.h" // Matlab
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdio.h>

#define CHRO_COUNT 24
#define CHRO_LEN 16 // En iyi 9 oluyor, görmek için 10 yapılabilir sayı
#define POINT_COUNT 250
#define ASCENDING 0
#define DESCENDING 1
#define BASE_X 50
#define BASE_Y 0
#define MAX_ITERATION 6000
#define MAX_ELITE_COUNT 1

extern double ITERATION_VALUES[MAX_ITERATION];
extern float POINTS_X[POINT_COUNT];
extern float POINTS_Y[POINT_COUNT];
extern int NODE_HEADS_FOR_CHRO[CHRO_COUNT][POINT_COUNT];

extern int START_POPULATION[CHRO_COUNT][CHRO_LEN];
extern double CHRO_FITNESS_VALUE[CHRO_COUNT];
extern int CUR_POPULATION[CHRO_COUNT][CHRO_LEN];
extern double CUR_FITNESS_VALUE[CHRO_COUNT];
extern int LAST_POPULATION[CHRO_COUNT][CHRO_LEN];
extern double LAST_FITNESS_VALUE[CHRO_COUNT];
extern int crIdArr[CHRO_COUNT];

double runCalcClusterHeadsAndTotalEnergy(int *h_out, float *h_in_x, float * h_in_y, int *h_in_ch, int arr_size_ch, int arr_size, int base_x, int base_y);

void sortArrayByFitness(double * fitness_arr, int pop_arr[CHRO_COUNT][CHRO_LEN], int way)
{
    double swap;
    int swap_pop[CHRO_LEN];
    for (int i = 0 ; i < CHRO_COUNT-1; i++)
    {
        for (int k = 0 ; k < CHRO_COUNT-1-i; k++)
        {
            if (way == ASCENDING)
            {
                if (fitness_arr[k] > fitness_arr[k+1]) /* For decreasing order use < */
                {
                    swap             = fitness_arr[k];
                    fitness_arr[k]   = fitness_arr[k+1];
                    fitness_arr[k+1] = swap;
                    for (int j = 0; j < CHRO_LEN; j++)
                    {
                        swap_pop[j]      = pop_arr[k][j];
                        pop_arr[k][j]    = pop_arr[k+1][j];
                        pop_arr[k+1][j]    = swap_pop[j];
                    }
                }
            }
            else if (way == DESCENDING)
            {
                if (fitness_arr[k] < fitness_arr[k+1])
                {
                    swap             = fitness_arr[k];
                    fitness_arr[k]   = fitness_arr[k+1];
                    fitness_arr[k+1] = swap;
                    for (int j = 0; j < CHRO_LEN; j++)
                    {
                        swap_pop[j]      = pop_arr[k][j];
                        pop_arr[k][j]    = pop_arr[k+1][j];
                        pop_arr[k+1][j]    = swap_pop[j];
                    }
                }
            }
        }
    }
}

void createRandomPopulation()
{
    for (int i = 0; i < CHRO_COUNT; i++)
    {
        crIdArr[i] = i;

        for (int k = 0; k < CHRO_LEN; k++)
        {
            LAST_POPULATION[i][k] = CUR_POPULATION[i][k] = START_POPULATION[i][k] = (rand() % POINT_COUNT);
        }
    }
    for (int i = 0; i < CHRO_COUNT; i++)
    {
        LAST_POPULATION[i][5] = CUR_POPULATION[i][5] = START_POPULATION[i][5] = -1;
    }
    sortArrayByFitness(LAST_FITNESS_VALUE, LAST_POPULATION, ASCENDING);
    std::random_shuffle(&crIdArr[0], &crIdArr[CHRO_COUNT]);
}

void avoidRepetitionChromosome(int index, int arr[CHRO_COUNT][CHRO_LEN])
{
    bool recursive = true;
    while(recursive)
    {
        for (int i = 0; i < CHRO_LEN; i++)
        {
            if (arr[index][i] == -1) continue;
            bool matching = false;
            for (int j = 0; (j < i) && (matching == false); j++)
            {
                if (arr[index][i] == arr[index][j])
                {
                    matching = true;
                    arr[index][j] = rand() % POINT_COUNT;
                }
            }
            if (matching) recursive = true;
            else recursive = false;
        }
    }
}

void calcLastGenerationsFitness()
{
    const int ARRAY_SIZE_POINTS = POINT_COUNT;
    const int ARRAY_SIZE_CH = CHRO_LEN;
    float *h_in_x = POINTS_X;
    float *h_in_y = POINTS_Y;
    int h_out[ARRAY_SIZE_POINTS];
    int h_in_ch[ARRAY_SIZE_CH];

    for (int i = 0; i < CHRO_COUNT; i++)
    {
        for (int k = 0; k < CHRO_LEN; k++)
        {
            h_in_ch[k] = LAST_POPULATION[i][k];
        }

        LAST_FITNESS_VALUE[i] = runCalcClusterHeadsAndTotalEnergy(h_out, h_in_x, h_in_y, h_in_ch, ARRAY_SIZE_CH, ARRAY_SIZE_POINTS, BASE_X, BASE_Y);
        for(int j = 0; j < ARRAY_SIZE_POINTS; j++)
        {
            NODE_HEADS_FOR_CHRO[i][j] = h_out[j];
        }
    }
    sortArrayByFitness(LAST_FITNESS_VALUE, LAST_POPULATION, ASCENDING);
}

void calcCurGenerationsFitness()
{
    const int ARRAY_SIZE_POINTS = POINT_COUNT;
    const int ARRAY_SIZE_CH = CHRO_LEN;
    float *h_in_x = POINTS_X;
    float *h_in_y = POINTS_Y;
    int h_out[ARRAY_SIZE_POINTS];
    int h_in_ch[ARRAY_SIZE_CH];

    for (int i = 0; i < CHRO_COUNT; i++)
    {
        for (int k = 0; k < CHRO_LEN; k++)
        {
            h_in_ch[k] = CUR_POPULATION[i][k];
        }
        CUR_FITNESS_VALUE[i] = runCalcClusterHeadsAndTotalEnergy(h_out, h_in_x, h_in_y, h_in_ch, ARRAY_SIZE_CH, ARRAY_SIZE_POINTS, BASE_X, BASE_Y);
    }
    sortArrayByFitness(CUR_FITNESS_VALUE, CUR_POPULATION, DESCENDING);
}

void crossingOverLastToCur()
{
    int first, second, cross_point = rand() % CHRO_LEN;
    for (int i = 0; i < CHRO_COUNT; i+=2)
    {
        first = crIdArr[i] = i;
        second = crIdArr[i+1];
        cross_point = rand() % CHRO_LEN;
        for(int j = 0; j < CHRO_LEN; j++)
        {
            if (j < cross_point)
            {
                CUR_POPULATION[i][j]= LAST_POPULATION[first][j];
                CUR_POPULATION[i+1][j] = LAST_POPULATION[second][j];
            }
            else
            {
                CUR_POPULATION[i][j]= LAST_POPULATION[second][j];
                CUR_POPULATION[i+1][j] = LAST_POPULATION[first][j];
            }
        }
        if (rand() % 1000 < 50) CUR_POPULATION[i][rand() % CHRO_LEN] = rand() % POINT_COUNT;//rand() % POINT_COUNT; // MUTATION 0.05
        avoidRepetitionChromosome(i, CUR_POPULATION);
        if (rand() % 1000 < 50) CUR_POPULATION[i+1][rand() % CHRO_LEN] = rand() % POINT_COUNT; // MUTATION 0.05
        avoidRepetitionChromosome(i+1, CUR_POPULATION);

        CUR_POPULATION[i][rand() % CHRO_LEN] = -1;
    }
}

void selectElitesFromCurToLast()
{
    for (int i = (CHRO_COUNT - MAX_ELITE_COUNT); i < CHRO_COUNT; i++)
    {
        LAST_FITNESS_VALUE[i] = CUR_FITNESS_VALUE[i];
        for (int k = 0; k < CHRO_LEN; k++)
        {
            LAST_POPULATION[i][k] = CUR_POPULATION[i][k];
        }
    }
}

void calcGeneration()
{
    crossingOverLastToCur();
    calcLastGenerationsFitness();
    calcCurGenerationsFitness();
    selectElitesFromCurToLast();
    calcLastGenerationsFitness(); // for Mutation Probability
    //sortArrayByFitness(LAST_FITNESS_VALUE, LAST_POPULATION, ASCENDING);
}

QString provideStream(QString file_name)
{
    QFile file1(file_name);
    if(!file1.open(QFile::ReadOnly | QFile::Text)) return "X";

    QTextStream qts1(&file1);
    qts1.setCodec("ISO-8859-9");
    QString str = qts1.readAll();
    file1.close();

    if(str.count("  ") > 0 || str.count("   ") > 0 || str.count("    ") > 0) return "X";

    return str;
}

QString givePoints(QString file_name)
{
    QString string_all = provideStream(file_name);
    if (string_all == "X") return "Can't open or corrupted file.";

    QStringList rows = string_all.split('\n');
    int row_count = rows.count()-1; // last row is empty
    QStringList words;

    for(int i = 0; i < row_count; i++)
    {
        words = rows[i].split(",");
        //kelime_sayisi = kelimeler.count();
        POINTS_X[i] = (float) (words[0].toInt() * 1.0);
        POINTS_Y[i] = (float) (words[1].toInt() * 1.0);
        //printf("%d,%d\n",words[0].toInt(),words[1].toInt());
    }
    return "Give Points success.";
}

void giveRandomPoints()
{
    for(int i = 0; i < POINT_COUNT; i++)
    {
        POINTS_X[i] = (float) (rand()%100 * 1.0);
        POINTS_Y[i] = (float) (rand()%100 * 1.0);
    }
}

#endif // CALCULATIONS_H

//double valsToSortDouble[13] = {1.4, 50.2, 5.11, -1.55, 301.521, 0.3301, 40.17, -18.0, 88.1, 30.44, -37.2, 3012.0, 49.2};
//int valsToSortInt[13] = {1, 50, 5, -1, 301, 0, 40, -18, 88, 30, -37, 3012, 49};
