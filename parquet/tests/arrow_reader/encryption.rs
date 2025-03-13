// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! This file contains an end to end test for modular encryption

use arrow_array::cast::AsArray;
use arrow_array::{types, RecordBatch};
use arrow_schema::ArrowError;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::encryption::decrypt::FileDecryptionProperties;
use parquet::errors::ParquetError;
use parquet::file::metadata::FileMetaData;
use std::fs::File;

#[test]
fn test_non_uniform_encryption_plaintext_footer() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_plaintext_footer.parquet.encrypted");
    let file = File::open(path).unwrap();

    // There is always a footer key even with a plaintext footer,
    // but this is used for signing the footer.
    let footer_key = "0123456789012345".as_bytes(); // 128bit/16
    let column_1_key = "1234567890123450".as_bytes();
    let column_2_key = "1234567890123451".as_bytes();

    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key.to_vec())
        .with_column_key("float_field", column_2_key.to_vec())
        .build()
        .unwrap();

    verify_encryption_test_file_read(file, decryption_properties);
}

#[test]
fn test_non_uniform_encryption_disabled_aad_storage() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path =
        format!("{testdata}/encrypt_columns_and_footer_disable_aad_storage.parquet.encrypted");
    let file = File::open(path.clone()).unwrap();

    let footer_key = "0123456789012345".as_bytes(); // 128bit/16
    let column_1_key = "1234567890123450".as_bytes();
    let column_2_key = "1234567890123451".as_bytes();

    // Can read successfully when providing the correct AAD prefix
    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key.to_vec())
        .with_column_key("float_field", column_2_key.to_vec())
        .with_aad_prefix("tester".as_bytes().to_vec())
        .build()
        .unwrap();

    verify_encryption_test_file_read(file, decryption_properties);

    // Using wrong AAD prefix should fail
    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key.to_vec())
        .with_column_key("float_field", column_2_key.to_vec())
        .with_aad_prefix("wrong_aad_prefix".as_bytes().to_vec())
        .build()
        .unwrap();

    let file = File::open(path.clone()).unwrap();
    let options = ArrowReaderOptions::default()
        .with_file_decryption_properties(decryption_properties.clone());
    let result = ArrowReaderMetadata::load(&file, options.clone());
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "Parquet error: Provided footer key and AAD were unable to decrypt parquet footer"
    );

    // Not providing any AAD prefix should fail as it isn't stored in the file
    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key.to_vec())
        .with_column_key("float_field", column_2_key.to_vec())
        .build()
        .unwrap();

    let file = File::open(path).unwrap();
    let options = ArrowReaderOptions::default()
        .with_file_decryption_properties(decryption_properties.clone());
    let result = ArrowReaderMetadata::load(&file, options.clone());
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "Parquet error: Provided footer key and AAD were unable to decrypt parquet footer"
    );
}

#[test]
fn test_non_uniform_encryption_plaintext_footer_without_decryption() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_plaintext_footer.parquet.encrypted");
    let file = File::open(&path).unwrap();

    let metadata = ArrowReaderMetadata::load(&file, Default::default()).unwrap();
    let file_metadata = metadata.metadata().file_metadata();

    assert_eq!(file_metadata.num_rows(), 50);
    assert_eq!(file_metadata.schema_descr().num_columns(), 8);
    assert_eq!(
        file_metadata.created_by().unwrap(),
        "parquet-cpp-arrow version 19.0.0-SNAPSHOT"
    );

    metadata.metadata().row_groups().iter().for_each(|rg| {
        assert_eq!(rg.num_columns(), 8);
        assert_eq!(rg.num_rows(), 50);
    });

    // Should be able to read unencrypted columns. Test reading one column.
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [1]);
    let record_reader = builder.with_projection(mask).build().unwrap();

    let mut row_count = 0;
    for batch in record_reader {
        let batch = batch.unwrap();
        row_count += batch.num_rows();

        let time_col = batch
            .column(0)
            .as_primitive::<types::Time32MillisecondType>();
        for (i, x) in time_col.iter().enumerate() {
            assert_eq!(x.unwrap(), i as i32);
        }
    }

    assert_eq!(row_count, file_metadata.num_rows() as usize);

    // Reading an encrypted column should fail
    let file = File::open(&path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [4]);
    let mut record_reader = builder.with_projection(mask).build().unwrap();

    match record_reader.next() {
        Some(Err(ArrowError::ParquetError(s))) => {
            assert!(s.contains("protocol error"));
        }
        _ => {
            panic!("Expected ArrowError::ParquetError");
        }
    };
}

#[test]
fn test_non_uniform_encryption() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_and_footer.parquet.encrypted");
    let file = File::open(path).unwrap();

    let footer_key = "0123456789012345".as_bytes(); // 128bit/16
    let column_1_key = "1234567890123450".as_bytes();
    let column_2_key = "1234567890123451".as_bytes();

    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key.to_vec())
        .with_column_key("float_field", column_2_key.to_vec())
        .build()
        .unwrap();

    verify_encryption_test_file_read(file, decryption_properties);
}

#[test]
fn test_uniform_encryption() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/uniform_encryption.parquet.encrypted");
    let file = File::open(path).unwrap();

    let key_code: &[u8] = "0123456789012345".as_bytes();
    let decryption_properties = FileDecryptionProperties::builder(key_code.to_vec())
        .build()
        .unwrap();

    verify_encryption_test_file_read(file, decryption_properties);
}

#[test]
fn test_decrypting_without_decryption_properties_fails() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/uniform_encryption.parquet.encrypted");
    let file = File::open(path).unwrap();

    let options = ArrowReaderOptions::default();
    let result = ArrowReaderMetadata::load(&file, options.clone());
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "Parquet error: Parquet file has an encrypted footer but no decryption properties were provided"
    );
}

#[test]
fn test_aes_ctr_encryption() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_and_footer_ctr.parquet.encrypted");
    let file = File::open(path).unwrap();

    let footer_key = "0123456789012345".as_bytes();
    let column_1_key = "1234567890123450".as_bytes();
    let column_2_key = "1234567890123451".as_bytes();

    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key.to_vec())
        .with_column_key("float_field", column_2_key.to_vec())
        .build()
        .unwrap();

    let options =
        ArrowReaderOptions::default().with_file_decryption_properties(decryption_properties);
    let metadata = ArrowReaderMetadata::load(&file, options);

    match metadata {
        Err(ParquetError::NYI(s)) => {
            assert!(s.contains("AES_GCM_CTR_V1"));
        }
        _ => {
            panic!("Expected ParquetError::NYI");
        }
    };
}

#[tokio::test]
#[cfg(feature = "encryption")]
async fn test_non_uniform_encryption_plaintext_footer_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_plaintext_footer.parquet.encrypted");
    let mut file = tokio::fs::File::open(&path).await.unwrap();

    // There is always a footer key even with a plaintext footer,
    // but this is used for signing the footer.
    let footer_key = "0123456789012345".as_bytes().to_vec(); // 128bit/16
    let column_1_key = "1234567890123450".as_bytes().to_vec();
    let column_2_key = "1234567890123451".as_bytes().to_vec();

    let decryption_properties = FileDecryptionProperties::builder(footer_key)
        .with_column_key("double_field", column_1_key)
        .with_column_key("float_field", column_2_key)
        .build()
        .unwrap();

    verify_encryption_test_file_read_async(&mut file, decryption_properties)
        .await
        .unwrap();
}

#[tokio::test]
#[cfg(feature = "encryption")]
async fn test_misspecified_encryption_keys_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_and_footer.parquet.encrypted");

    // There is always a footer key even with a plaintext footer,
    // but this is used for signing the footer.
    let footer_key = "0123456789012345".as_bytes(); // 128bit/16
    let column_1_key = "1234567890123450".as_bytes();
    let column_2_key = "1234567890123451".as_bytes();

    // read file with keys and check for expected error message
    async fn check_for_error(
        expected_message: &str,
        path: &String,
        footer_key: &[u8],
        column_1_key: &[u8],
        column_2_key: &[u8],
    ) {
        let mut file = tokio::fs::File::open(&path).await.unwrap();

        let mut decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec());

        if !column_1_key.is_empty() {
            decryption_properties =
                decryption_properties.with_column_key("double_field", column_1_key.to_vec());
        }

        if !column_2_key.is_empty() {
            decryption_properties =
                decryption_properties.with_column_key("float_field", column_2_key.to_vec());
        }

        let decryption_properties = decryption_properties.build().unwrap();

        match verify_encryption_test_file_read_async(&mut file, decryption_properties).await {
            Ok(_) => {
                panic!("did not get expected error")
            }
            Err(e) => {
                assert_eq!(e.to_string(), expected_message);
            }
        }
    }

    // Too short footer key
    check_for_error(
        "Parquet error: Invalid footer key. Failed to create AES key",
        &path,
        "bad_pwd".as_bytes(),
        column_1_key,
        column_2_key,
    )
    .await;

    // Wrong footer key
    check_for_error(
        "Parquet error: Provided footer key and AAD were unable to decrypt parquet footer",
        &path,
        "1123456789012345".as_bytes(),
        column_1_key,
        column_2_key,
    )
    .await;

    // Missing column key
    check_for_error("Parquet error: Unable to decrypt column 'double_field', perhaps the column key is wrong or missing?",
                    &path, footer_key, "".as_bytes(), column_2_key).await;

    // Too short column key
    check_for_error(
        "Parquet error: Failed to create AES key",
        &path,
        footer_key,
        "abc".as_bytes(),
        column_2_key,
    )
    .await;

    // Wrong column key
    check_for_error("Parquet error: Unable to decrypt column 'double_field', perhaps the column key is wrong or missing?",
                    &path, footer_key, "1123456789012345".as_bytes(), column_2_key).await;

    // Mixed up keys
    check_for_error("Parquet error: Unable to decrypt column 'float_field', perhaps the column key is wrong or missing?",
                    &path, footer_key, column_2_key, column_1_key).await;
}

#[tokio::test]
async fn test_non_uniform_encryption_plaintext_footer_without_decryption_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_plaintext_footer.parquet.encrypted");
    let mut file = tokio::fs::File::open(&path).await.unwrap();

    let metadata = ArrowReaderMetadata::load_async(&mut file, Default::default())
        .await
        .unwrap();
    let file_metadata = metadata.metadata().file_metadata();

    assert_eq!(file_metadata.num_rows(), 50);
    assert_eq!(file_metadata.schema_descr().num_columns(), 8);
    assert_eq!(
        file_metadata.created_by().unwrap(),
        "parquet-cpp-arrow version 19.0.0-SNAPSHOT"
    );

    metadata.metadata().row_groups().iter().for_each(|rg| {
        assert_eq!(rg.num_columns(), 8);
        assert_eq!(rg.num_rows(), 50);
    });

    // Should be able to read unencrypted columns. Test reading one column.
    let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [1]);
    let record_reader = builder.with_projection(mask).build().unwrap();
    let record_batches = record_reader.try_collect::<Vec<_>>().await.unwrap();

    let mut row_count = 0;
    for batch in record_batches {
        let batch = batch;
        row_count += batch.num_rows();

        let time_col = batch
            .column(0)
            .as_primitive::<types::Time32MillisecondType>();
        for (i, x) in time_col.iter().enumerate() {
            assert_eq!(x.unwrap(), i as i32);
        }
    }

    assert_eq!(row_count, file_metadata.num_rows() as usize);

    // Reading an encrypted column should fail
    let file = tokio::fs::File::open(&path).await.unwrap();
    let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [4]);
    let mut record_reader = builder.with_projection(mask).build().unwrap();

    match record_reader.next().await {
        Some(Err(ParquetError::ArrowError(s))) => {
            assert!(s.contains("protocol error"));
        }
        _ => {
            panic!("Expected ArrowError::ParquetError");
        }
    };
}

#[tokio::test]
#[cfg(feature = "encryption")]
async fn test_non_uniform_encryption_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_and_footer.parquet.encrypted");
    let mut file = tokio::fs::File::open(&path).await.unwrap();

    let footer_key = "0123456789012345".as_bytes().to_vec(); // 128bit/16
    let column_1_key = "1234567890123450".as_bytes().to_vec();
    let column_2_key = "1234567890123451".as_bytes().to_vec();

    let decryption_properties = FileDecryptionProperties::builder(footer_key.to_vec())
        .with_column_key("double_field", column_1_key)
        .with_column_key("float_field", column_2_key)
        .build()
        .unwrap();

    verify_encryption_test_file_read_async(&mut file, decryption_properties)
        .await
        .unwrap();
}

#[tokio::test]
#[cfg(feature = "encryption")]
async fn test_uniform_encryption_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/uniform_encryption.parquet.encrypted");
    let mut file = tokio::fs::File::open(&path).await.unwrap();

    let key_code: &[u8] = "0123456789012345".as_bytes();
    let decryption_properties = FileDecryptionProperties::builder(key_code.to_vec())
        .build()
        .unwrap();

    verify_encryption_test_file_read_async(&mut file, decryption_properties)
        .await
        .unwrap();
}

#[tokio::test]
#[cfg(feature = "encryption")]
async fn test_aes_ctr_encryption_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/encrypt_columns_and_footer_ctr.parquet.encrypted");
    let mut file = tokio::fs::File::open(&path).await.unwrap();

    let footer_key = "0123456789012345".as_bytes().to_vec();
    let column_1_key = "1234567890123450".as_bytes().to_vec();
    let column_2_key = "1234567890123451".as_bytes().to_vec();

    let decryption_properties = FileDecryptionProperties::builder(footer_key)
        .with_column_key("double_field", column_1_key)
        .with_column_key("float_field", column_2_key)
        .build()
        .unwrap();

    let options = ArrowReaderOptions::new().with_file_decryption_properties(decryption_properties);
    let metadata = ArrowReaderMetadata::load_async(&mut file, options).await;

    match metadata {
        Err(ParquetError::NYI(s)) => {
            assert!(s.contains("AES_GCM_CTR_V1"));
        }
        _ => {
            panic!("Expected ParquetError::NYI");
        }
    };
}

#[tokio::test]
#[cfg(feature = "encryption")]
async fn test_decrypting_without_decryption_properties_fails_async() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/uniform_encryption.parquet.encrypted");
    let mut file = tokio::fs::File::open(&path).await.unwrap();

    let options = ArrowReaderOptions::new();
    let result = ArrowReaderMetadata::load_async(&mut file, options).await;
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "Parquet error: Parquet file has an encrypted footer but no decryption properties were provided"
    );
}

// *******  Utilities ********

fn verify_encryption_test_file_read(file: File, decryption_properties: FileDecryptionProperties) {
    let options = ArrowReaderOptions::default()
        .with_file_decryption_properties(decryption_properties.clone());
    let metadata = ArrowReaderMetadata::load(&file, options.clone()).unwrap();
    let file_metadata = metadata.metadata().file_metadata();

    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options).unwrap();
    let record_reader = builder.build().unwrap();
    let record_batches = record_reader
        .map(|x| x.unwrap())
        .collect::<Vec<RecordBatch>>();

    verify_encryption_test_data(record_batches, file_metadata.clone(), metadata);
}

async fn verify_encryption_test_file_read_async(
    file: &mut tokio::fs::File,
    decryption_properties: FileDecryptionProperties,
) -> Result<(), ParquetError> {
    let options = ArrowReaderOptions::new().with_file_decryption_properties(decryption_properties);

    let metadata = ArrowReaderMetadata::load_async(file, options.clone()).await?;
    let arrow_reader_metadata = ArrowReaderMetadata::load_async(file, options).await?;
    let file_metadata = metadata.metadata().file_metadata();

    let record_reader = ParquetRecordBatchStreamBuilder::new_with_metadata(
        file.try_clone().await?,
        arrow_reader_metadata.clone(),
    )
    .build()?;
    let record_batches = record_reader.try_collect::<Vec<_>>().await?;

    verify_encryption_test_data(record_batches, file_metadata.clone(), metadata);
    Ok(())
}

/// Tests reading an encrypted file from the parquet-testing repository
fn verify_encryption_test_data(
    record_batches: Vec<RecordBatch>,
    file_metadata: FileMetaData,
    metadata: ArrowReaderMetadata,
) {
    assert_eq!(file_metadata.num_rows(), 50);
    assert_eq!(file_metadata.schema_descr().num_columns(), 8);

    metadata.metadata().row_groups().iter().for_each(|rg| {
        assert_eq!(rg.num_columns(), 8);
        assert_eq!(rg.num_rows(), 50);
    });

    let mut row_count = 0;
    for batch in record_batches {
        let batch = batch;
        row_count += batch.num_rows();

        let bool_col = batch.column(0).as_boolean();
        let time_col = batch
            .column(1)
            .as_primitive::<types::Time32MillisecondType>();
        let list_col = batch.column(2).as_list::<i32>();
        let timestamp_col = batch
            .column(3)
            .as_primitive::<types::TimestampNanosecondType>();
        let f32_col = batch.column(4).as_primitive::<types::Float32Type>();
        let f64_col = batch.column(5).as_primitive::<types::Float64Type>();
        let binary_col = batch.column(6).as_binary::<i32>();
        let fixed_size_binary_col = batch.column(7).as_fixed_size_binary();

        for (i, x) in bool_col.iter().enumerate() {
            assert_eq!(x.unwrap(), i % 2 == 0);
        }
        for (i, x) in time_col.iter().enumerate() {
            assert_eq!(x.unwrap(), i as i32);
        }
        for (i, list_item) in list_col.iter().enumerate() {
            let list_item = list_item.unwrap();
            let list_item = list_item.as_primitive::<types::Int64Type>();
            assert_eq!(list_item.len(), 2);
            assert_eq!(list_item.value(0), ((i * 2) * 1000000000000) as i64);
            assert_eq!(list_item.value(1), ((i * 2 + 1) * 1000000000000) as i64);
        }
        for x in timestamp_col.iter() {
            assert!(x.is_some());
        }
        for (i, x) in f32_col.iter().enumerate() {
            assert_eq!(x.unwrap(), i as f32 * 1.1f32);
        }
        for (i, x) in f64_col.iter().enumerate() {
            assert_eq!(x.unwrap(), i as f64 * 1.1111111f64);
        }
        for (i, x) in binary_col.iter().enumerate() {
            assert_eq!(x.is_some(), i % 2 == 0);
            if let Some(x) = x {
                assert_eq!(&x[0..7], b"parquet");
            }
        }
        for (i, x) in fixed_size_binary_col.iter().enumerate() {
            assert_eq!(x.unwrap(), &[i as u8; 10]);
        }
    }

    assert_eq!(row_count, file_metadata.num_rows() as usize);
}
