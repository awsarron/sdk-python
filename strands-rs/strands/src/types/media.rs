//! Media content blocks (image, video, document) and their sources.
//!
//! Ports the media types from the TypeScript SDK's `types/media.ts`. Raw byte
//! sources are base64-encoded on the wire, matching the TypeScript SDK, which
//! encodes `Uint8Array` as a base64 string in `toJSON()`.

use serde::{Deserialize, Serialize};

use crate::types::base64_bytes;

/// A reference to an object stored in Amazon S3.
///
/// Carries a `type: "s3"` discriminator on the wire, matching the TypeScript
/// `S3LocationData` shape so serialized media round-trips across SDKs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct S3Location {
    /// Discriminator; always `"s3"`.
    #[serde(rename = "type")]
    pub kind: S3LocationKind,
    /// The S3 URI of the object.
    pub uri: String,
    /// The AWS account ID that owns the bucket, when a cross-account reference is required.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket_owner: Option<String>,
}

/// Discriminator value for [`S3Location`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum S3LocationKind {
    /// The only supported location kind.
    S3,
}

/// The source of media content: raw bytes, an S3 location, or a URL.
///
/// Mirrors the TypeScript source union: `{ bytes }`, `{ location }`, `{ url }`.
/// Serde external tagging selects the variant by JSON key, matching how the
/// TypeScript `fromJSON` discriminates via `'bytes' in source` / `'location' in source`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MediaSource {
    /// Raw bytes, base64-encoded on the wire under the `bytes` key.
    Bytes(#[serde(with = "base64_bytes")] Vec<u8>),
    /// A reference to an object in S3, under the `location` key.
    Location(S3Location),
    /// A remote URL, under the `url` key. Supported for images in the TypeScript SDK.
    Url(String),
}

/// An image content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageBlock {
    /// The image format (e.g. `png`, `jpeg`, `gif`, `webp`).
    pub format: String,
    /// The image data source.
    pub source: MediaSource,
}

/// A video content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoBlock {
    /// The video format (e.g. `mp4`, `mov`, `webm`).
    pub format: String,
    /// The video data source.
    pub source: MediaSource,
}

/// A document content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentBlock {
    /// The document format (e.g. `pdf`, `txt`, `md`, `csv`).
    pub format: String,
    /// The document name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The document data source.
    pub source: MediaSource,
}
