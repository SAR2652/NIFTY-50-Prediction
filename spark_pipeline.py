import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql.functions import lag
from pyspark.sql.window import Window

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, serialize

from elephas.ml_model import ElephasEstimator

def dl_pipeline_fit_score_results(dl_pipeline,
                                  train_data,
                                  test_data,
                                  label='target_close'):
    
    fit_dl_pipeline = dl_pipeline.fit(train_data)
    pred_train = fit_dl_pipeline.transform(train_data)
    pred_test = fit_dl_pipeline.transform(test_data)
    
    pnl_train = pred_train.select(label, "prediction")
    pnl_test = pred_test.select(label, "prediction")
    
    pred_and_label_train = pnl_train.rdd.map(lambda row: (row[label], row['prediction']))
    pred_and_label_test = pnl_test.rdd.map(lambda row: (row[label], row['prediction']))
    
    metrics_train = RegressionMetrics(pred_and_label_train)
    metrics_test = RegressionMetrics(pred_and_label_test)
    
    train_rmse = round(metrics_train.rootMeanSquaredError(), 4)
    train_r2 = round(metrics_train.r2(), 4)
    
    logging.info(f"Training Data RMSE = {train_rmse}")
    logging.info(f"Training Data R^2 = {train_r2}")
    
    test_rmse = round(metrics_test.rootMeanSquaredError(), 4)
    test_r2 = round(metrics_test.r2(), 4)
    
    logging.info(f"Test Data RMSE = {test_rmse}")
    logging.info(f"Test Data R^2 = {test_r2}")


def main():
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='pipeline.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    spark = SparkSession.builder.master("local[1]").appName("SparkMLTrainer")\
                                .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.options(header='True', delimiter=',', inferSchema='True').csv("NIFTY_5_minute_with_indicators_.csv", enforceSchema=True)
    # df.printSchema()
    df.show(5)

    df.drop("date", "volume").printSchema()

    # define the lag window
    window = Window.orderBy("close")

    for col in df.columns:
        df = df.withColumn(f"Lag_{col}", lag(col).over(window))

    # Highly correlated features obtained from EDA
    high_corr_features = ('close', 'Lag_close', 'Lag_high', 'Lag_low', 'Lag_open', 'Lag_sma5', 'Lag_sma10', 'Lag_sma15', 'Lag_sma20', 'Lag_ema5', 'Lag_ema10', 'Lag_ema15', 'Lag_ema20', 'Lag_upperband', 'Lag_middleband', 'Lag_lowerband', 'Lag_HT_TRENDLINE', 'Lag_KAMA10', 'Lag_KAMA20', 'Lag_KAMA30', 'Lag_TRIMA5', 'Lag_TRIMA10', 'Lag_TRIMA20', 'Lag_TYPPRICE')

    # Decide columns to be dropped
    drop_cols = [col for col in df.columns if col not in high_corr_features]
    drop_cols = tuple(drop_cols)
    df = df.drop(*drop_cols)
    df.show(5)
    df = df.na.drop('any')
    # n_rows = df.count()
    # train_size = int(0.6 * n_rows)
    # val_test_size = n_rows - train_size
    df.toPandas().to_csv('data.csv', index=False)

    # train_df_us = df.limit(train_size)
    # val_test_df_rows_us = df.tail(val_test_size)
    # val_test_df = spark.createDataFrame(val_test_df_rows_us)
    # val_df_us = val_test_df.limit(val_test_size // 2)
    # test_df_rows_us = val_test_df.tail(val_test_size // 2)
    # test_df_us = spark.createDataFrame(test_df_rows_us)

    # Other CSV options
    # train_df_us.toPandas().to_csv("train_unscaled.csv", index=False)
    # val_df_us.toPandas().to_csv("val_unscaled.csv", index=False)
    # test_df_us.toPandas().to_csv("test_unscaled.csv", index=False)
    
    high_corr_features = list(high_corr_features)
    # high_corr_features.remove('close')

    stages = []

    target_assembler = VectorAssembler(inputCols=['close'],
                                    outputCol='scaled_target')
    t_scaler = MinMaxScaler(inputCol='scaled_target', outputCol='target_close')
    # t_scaler = MinMaxScaler(inputCol='close', outputCol='target_close')
    # scaler_model = t_scaler.fit(df)
    # scaled_df = scaler_model.transform(df)

    stages += [target_assembler, t_scaler]
    

    unscaled_feature_assembler = VectorAssembler(inputCols=high_corr_features,
                                                outputCol="high_corr_features")
    scaler = MinMaxScaler(inputCol="high_corr_features",
                          outputCol="scaled_features")

    stages += [unscaled_feature_assembler, scaler]

    # Set Pipeline
    pipeline = Pipeline(stages=stages)

    # Fit Pipeline to Data
    pipeline_model = pipeline.fit(df)

    # Transform Data using Fitted Pipeline
    df_transform = pipeline_model.transform(df)

    # select features
    df_transform_fin = df_transform.select('scaled_features', 'target_close')

    # df_transform_fin.show(5)

    n_rows = df_transform_fin.count()
    train_size = int(0.8 * n_rows)
    test_size = n_rows - train_size
    logging.info(f"Train Size = {train_size}")
    logging.info(f"Test Size = {test_size}")

    # select first 80% rows
    train_df = df_transform_fin.limit(train_size)
    logging.info('Training Size')
    train_df.show(5)

    # select last 40% rows
    test_df_rows = df_transform_fin.tail(test_size)
    test_df = spark.createDataFrame(test_df_rows)
    logging.info('Validation + Test Size')
    test_df.show(5)

    # select first 50% of last 40% remaining rows of the original data
    # val_df = val_test_df.limit(val_test_size // 2)
    
    # select last 50% of last 40% remaining rows of the original data
    test_df_rows = test_df.tail(test_size)
    test_df = spark.createDataFrame(test_df_rows)

    # Number of Inputs or Input Dimensions
    # logging.info(train_df.select("scaled_features").first()[0])
    input_dim = len(train_df.select("scaled_features").first()[0])
    logging.info(f'Input Dimensions = {input_dim}')

    # convert pipeline
    # train_df_pd = train_df.toPandas()
    # val_df_pd = val_df.toPandas()
    # test_df_pd = test_df.toPandas()

    # tf_shape = train_df_pd['scaled_features'].values.shape
    # logging.info(f'Train Features Shape = {tf_shape}')
    # tt_shape = train_df_pd['target_close'].values.shape
    # logging.info(f'Train Target Shape = {tt_shape}')
    # vf_shape = val_df_pd['scaled_features'].values.shape
    # logging.info(f'Validation Features Shape = {vf_shape}')
    # vt_shape = val_df_pd['target_close'].values.shape
    # logging.info(f'Validation Target Shape = {vt_shape}')
    # logging.info(f'Test Features Shapee = {test_df_pd['scaled_features'].values.shape}')
    # logging.info(f'Test Target Shape = {test_df_pd['target_close'].values.shape}')
    # X_train = np.array(train_df_pd['scaled_features'].tolist())
    # X_val = np.array(val_df_pd['scaled_features'].tolist())
    # X_test = np.array(test_df_pd['scaled_features'].tolist())
    # y_train = np.array(train_df_pd['target_close'].tolist())
    # y_val = np.array(val_df_pd['target_close'].tolist())
    # y_test = np.array(test_df_pd['target_close'].tolist())
    # with open('X_train.npy', 'wb') as f:
    #     np.save(f, X_train)
    # with open('X_val.npy', 'wb') as f:
    #     np.save(f, X_val)
    # with open('X_test.npy', 'wb') as f:
    #     np.save(f, X_test)
    # with open('y_train.npy', 'wb') as f:
    #     np.save(f, y_train)
    # with open('y_val.npy', 'wb') as f:
    #     np.save(f, y_val)
    # with open('y_test.npy', 'wb') as f:
    #     np.save(f, y_test)
    # logging.info(y_test.shape)
    # logging.info(y_train.shape)
    # logging.info(y_test[0])
    # logging.info(X_train.shape)
    # model = Sequential()
    # model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1, activation='linear'))

    # model.compile(loss='mean_squared_error', optimizer='adam')

    # model.summary()

    # # Set and Serialize Optimizer
    # optimizer_conf = Adam(learning_rate=0.001)
    # opt_conf = serialize(optimizer_conf)

    # # Initialize SparkML Estimator and Get Settings
    # estimator = ElephasEstimator()
    # estimator.setFeaturesCol("scaled_features")
    # estimator.setLabelCol("target_close")
    # estimator.set_keras_model_config(model.to_json())
    # estimator.set_num_workers(1)
    # estimator.set_epochs(25) 
    # estimator.set_batch_size(32)
    # estimator.set_verbosity(1)
    # estimator.set_validation_split(0.1)
    # estimator.set_optimizer_config(opt_conf)
    # estimator.set_mode("synchronous")
    # estimator.set_loss("mean_squared_error")
    # estimator.set_metrics(['mean_squared_error', 'mean_absolute_error'])
    # estimator.set_categorical_labels(False)

    # # Create Deep Learning Pipeline
    # dl_pipeline = Pipeline(stages=[estimator])

    # dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
    #                               train_data=train_df,
    #                               test_data=test_df,
    #                               label='target_close');

if __name__ == '__main__':
    main()