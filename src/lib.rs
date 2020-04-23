use crate::header::{ElementDecl, PropertyDecl, PropertyType, Header, Format};
use nom::{IResult, Parser, Err};
use smallvec::SmallVec;
use std::collections::HashMap;
use nom::error::{ParseError, ErrorKind};
use nom::lib::std::ops::RangeFrom;
use nom::sequence::terminated;
use nom::character::complete::multispace0;

pub mod header;

#[derive(Debug, PartialEq)]
pub struct PlyData {
    pub elements: HashMap<String, ElementData>,
}

impl PlyData {
    pub fn parse_with_header<'a>(header: &'_ Header, input: &'a [u8]) -> IResult<&'a [u8], Self> {
        let elements = Vec::with_capacity(header.elements.len());
        let (input, elements) = header.elements.iter().try_fold((input, elements), |(input, mut elements), decl| {
            let (input, el) = match header.format {
                Format::Ascii => ElementData::parse_element::<Ascii>(decl, input)?,
                Format::BinaryLittleEndian => ElementData::parse_element::<BinaryLittleEndian>(decl, input)?,
                Format::BinaryBigEndian => ElementData::parse_element::<BinaryBigEndian>(decl, input)?,
            };
            elements.push(el);
            Ok((input, elements))
        })?;

        let elements = header.elements.iter().zip(elements.into_iter())
            .map(|(decl, el)| (decl.name.clone(), el))
            .collect();
        let data = Self {
            elements
        };
        Ok((input, data))
    }

    pub fn parse_complete(input: &[u8]) -> Result<Self, Err<(&[u8], ErrorKind)>> {
        let (input, header) = header::header(input)?;
        let (rest, data) = Self::parse_with_header(&header, input)?;
        Ok(data)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementData {
    pub properties: HashMap<String, PropertyData>,
}

impl ElementData {
    pub fn parse_element<'a, P: ParseProperty>(decl: &'_ ElementDecl, input: &'a [u8]) -> IResult<&'a [u8], Self> {
        let mut property_vecs: Vec<_> = decl
            .properties
            .iter()
            .map(|p| PropertyData::with_capacity(decl.count, p))
            .collect();

        let mut input = input;
        for _ in 0..decl.count {
            for (prop_data, prop_decl) in property_vecs.iter_mut().zip(decl.properties.iter()) {
                let (rest, _) =  Self::parse_one_property::<P>(input, prop_decl, prop_data)?;
                input = rest;
            }
        }
        let properties = decl.properties.iter().zip(property_vecs.into_iter())
            .map(|(decl, prop)| (decl.name().to_string(), prop))
            .collect();
        let element = Self {
            properties
        };
        Ok((input, element))
    }

    fn parse_one_property<'a, P: ParseProperty>(input: &'a [u8], decl: &PropertyDecl, prop_data: &mut PropertyData) -> IResult<&'a [u8], ()> {
        let input = match decl {
            PropertyDecl::Single { ty, .. } => {
                match (ty, prop_data) {
                    (PropertyType::Char, PropertyData::Char(v)) => {
                        let (input, prop) = P::char(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Uchar, PropertyData::Uchar(v)) => {
                        let (input, prop) = P::uchar(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Short, PropertyData::Short(v)) => {
                        let (input, prop) = P::short(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Ushort, PropertyData::Ushort(v)) => {
                        let (input, prop) = P::ushort(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Int, PropertyData::Int(v)) => {
                        let (input, prop) = P::int(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Uint, PropertyData::Uint(v)) => {
                        let (input, prop) = P::uint(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Float, PropertyData::Float(v)) => {
                        let (input, prop) = P::float(input)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Double, PropertyData::Double(v)) => {
                        let (input, prop) = P::double(input)?;
                        v.push(prop);
                        input
                    },
                    _ => unreachable!()
                }
            },
            PropertyDecl::List { ty, length_ty, .. } => {
                let (input, list_length) = Self::parse_list_length::<P>(input, *length_ty)?;
                match (ty, prop_data) {
                    (PropertyType::Char, PropertyData::ListChar(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::char)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Uchar, PropertyData::ListUchar(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::uchar)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Short, PropertyData::ListShort(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::short)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Ushort, PropertyData::ListUshort(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::ushort)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Int, PropertyData::ListInt(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::int)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Uint, PropertyData::ListUint(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::uint)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Float, PropertyData::ListFloat(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::float)?;
                        v.push(prop);
                        input
                    },
                    (PropertyType::Double, PropertyData::ListDouble(v)) => {
                        let (input, prop) = parse_list(input, list_length, P::double)?;
                        v.push(prop);
                        input
                    },
                    _ => unreachable!()
                }
            },
        };
        Ok((input, ()))
    }

    fn parse_list_length<P: ParseProperty>(input: &[u8], ty: PropertyType) -> IResult<&[u8], usize> {
        match ty {
            PropertyType::Char => (P::char).map(|n| n as usize).parse(input),
            PropertyType::Uchar => P::uchar.map(|n| n as usize).parse(input),
            PropertyType::Short => P::short.map(|n| n as usize).parse(input),
            PropertyType::Ushort => P::ushort.map(|n| n as usize).parse(input),
            PropertyType::Int => P::int.map(|n| n as usize).parse(input),
            PropertyType::Uint => P::uint.map(|n| n as usize).parse(input),
            PropertyType::Float => P::float.map(|n| n as usize).parse(input),
            PropertyType::Double => P::double.map(|n| n as usize).parse(input),
        }
    }

}

fn parse_list<F, T>(
    input: &[u8], count: usize, parser: F
) -> IResult<&[u8], SmallVec<[T; 4]>>
    where
        F: Fn(&[u8]) -> IResult<&[u8], T>,
{
    let mut sv = SmallVec::with_capacity(count);
    let mut input = input;
    for _ in 0..count {
        let (rest, prop) = parser(input)?;
        input = rest;
        sv.push(prop);
    }
    Ok((input, sv))
}

#[derive(Debug, Clone, PartialEq)]
pub enum PropertyData {
    Char(Vec<i8>),
    Uchar(Vec<u8>),
    Short(Vec<i16>),
    Ushort(Vec<u16>),
    Int(Vec<i32>),
    Uint(Vec<u32>),
    Float(Vec<f32>),
    Double(Vec<f64>),

    ListChar(Vec<SmallVec<[i8; 4]>>),
    ListUchar(Vec<SmallVec<[u8; 4]>>),
    ListShort(Vec<SmallVec<[i16; 4]>>),
    ListUshort(Vec<SmallVec<[u16; 4]>>),
    ListInt(Vec<SmallVec<[i32; 4]>>),
    ListUint(Vec<SmallVec<[u32; 4]>>),
    ListFloat(Vec<SmallVec<[f32; 4]>>),
    ListDouble(Vec<SmallVec<[f64; 4]>>),
}

impl PropertyData {
    fn with_capacity(n: usize, decl: &PropertyDecl) -> Self {
        use PropertyData::*;
        match decl {
            PropertyDecl::Single { ty, .. } => match ty {
                PropertyType::Char => Char(Vec::with_capacity(n)),
                PropertyType::Uchar => Uchar(Vec::with_capacity(n)),
                PropertyType::Short => Short(Vec::with_capacity(n)),
                PropertyType::Ushort => Ushort(Vec::with_capacity(n)),
                PropertyType::Int => Int(Vec::with_capacity(n)),
                PropertyType::Uint => Uint(Vec::with_capacity(n)),
                PropertyType::Float => Float(Vec::with_capacity(n)),
                PropertyType::Double => Double(Vec::with_capacity(n)),
            },
            PropertyDecl::List { ty, .. } => match ty {
                PropertyType::Char => ListChar(Vec::with_capacity(n)),
                PropertyType::Uchar => ListUchar(Vec::with_capacity(n)),
                PropertyType::Short => ListShort(Vec::with_capacity(n)),
                PropertyType::Ushort => ListUshort(Vec::with_capacity(n)),
                PropertyType::Int => ListInt(Vec::with_capacity(n)),
                PropertyType::Uint => ListUint(Vec::with_capacity(n)),
                PropertyType::Float => ListFloat(Vec::with_capacity(n)),
                PropertyType::Double => ListDouble(Vec::with_capacity(n)),
            },
        }
    }
}

pub trait ParseProperty {
    fn char(input: &[u8]) -> IResult<&[u8], i8>;
    fn uchar(input: &[u8]) -> IResult<&[u8], u8>;
    fn short(input: &[u8]) -> IResult<&[u8], i16>;
    fn ushort(input: &[u8]) -> IResult<&[u8], u16>;
    fn int(input: &[u8]) -> IResult<&[u8], i32>;
    fn uint(input: &[u8]) -> IResult<&[u8], u32>;
    fn float(input: &[u8]) -> IResult<&[u8], f32>;
    fn double(input: &[u8]) -> IResult<&[u8], f64>;
}

// TODO how to handle streaming
fn decimal_number<I, T, E: ParseError<I>>(input: I) -> IResult<I, T, E>
where
    I: nom::AsBytes + nom::InputLength + nom::Slice<RangeFrom<usize>>,
    T: lexical::FromLexical
{
    match T::from_lexical_partial(input.as_bytes()) {
        Ok((value, processed)) => {
            Ok((input.slice(processed..), value))
        },
        Err(_) => Err(nom::Err::Error(E::from_error_kind(input, ErrorKind::ParseTo)))
    }
}

pub struct Ascii;
pub struct BinaryLittleEndian;
pub struct BinaryBigEndian;

impl ParseProperty for Ascii {
    fn char(input: &[u8]) -> IResult<&[u8], i8> {
        terminated(decimal_number, multispace0)(input)
    }

    fn uchar(input: &[u8]) -> IResult<&[u8], u8> {
        terminated(decimal_number, multispace0)(input)
    }

    fn short(input: &[u8]) -> IResult<&[u8], i16> {
        terminated(decimal_number, multispace0)(input)
    }

    fn ushort(input: &[u8]) -> IResult<&[u8], u16> {
        terminated(decimal_number, multispace0)(input)
    }

    fn int(input: &[u8]) -> IResult<&[u8], i32> {
        terminated(decimal_number, multispace0)(input)
    }

    fn uint(input: &[u8]) -> IResult<&[u8], u32> {
        terminated(decimal_number, multispace0)(input)
    }

    fn float(input: &[u8]) -> IResult<&[u8], f32> {
        terminated(decimal_number, multispace0)(input)
    }

    fn double(input: &[u8]) -> IResult<&[u8], f64> {
        terminated(decimal_number, multispace0)(input)
    }
}

impl ParseProperty for BinaryLittleEndian {
    fn char(input: &[u8]) -> IResult<&[u8], i8> {
        unimplemented!()
    }

    fn uchar(input: &[u8]) -> IResult<&[u8], u8> {
        unimplemented!()
    }

    fn short(input: &[u8]) -> IResult<&[u8], i16> {
        unimplemented!()
    }

    fn ushort(input: &[u8]) -> IResult<&[u8], u16> {
        unimplemented!()
    }

    fn int(input: &[u8]) -> IResult<&[u8], i32> {
        unimplemented!()
    }

    fn uint(input: &[u8]) -> IResult<&[u8], u32> {
        unimplemented!()
    }

    fn float(input: &[u8]) -> IResult<&[u8], f32> {
        unimplemented!()
    }

    fn double(input: &[u8]) -> IResult<&[u8], f64> {
        unimplemented!()
    }
}

impl ParseProperty for BinaryBigEndian {
    fn char(input: &[u8]) -> IResult<&[u8], i8> {
        unimplemented!()
    }

    fn uchar(input: &[u8]) -> IResult<&[u8], u8> {
        unimplemented!()
    }

    fn short(input: &[u8]) -> IResult<&[u8], i16> {
        unimplemented!()
    }

    fn ushort(input: &[u8]) -> IResult<&[u8], u16> {
        unimplemented!()
    }

    fn int(input: &[u8]) -> IResult<&[u8], i32> {
        unimplemented!()
    }

    fn uint(input: &[u8]) -> IResult<&[u8], u32> {
        unimplemented!()
    }

    fn float(input: &[u8]) -> IResult<&[u8], f32> {
        unimplemented!()
    }

    fn double(input: &[u8]) -> IResult<&[u8], f64> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn test_basic_ascii() {
        let file = include_bytes!("test_files/simple_ascii.ply");

        let x = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let z = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let indices: Vec<SmallVec<[i32; 4]>> = vec![
            smallvec![0, 1, 2, 3],
            smallvec![7, 6, 5, 4],
            smallvec![0, 4, 5, 1],
            smallvec![1, 5, 6, 2],
            smallvec![2, 6, 7, 3],
            smallvec![3, 7, 4, 0],
        ];

        let mut properties = HashMap::new();
        properties.insert("x".to_string(), PropertyData::Float(x));
        properties.insert("y".to_string(), PropertyData::Float(y));
        properties.insert("z".to_string(), PropertyData::Float(z));
        let mut elements = HashMap::new();
        elements.insert("vertex".to_string(), ElementData { properties });
        let mut properties = HashMap::new();
        properties.insert("vertex_index".to_string(), PropertyData::ListInt(indices));
        elements.insert("face".to_string(), ElementData { properties });

        let data = PlyData::parse_complete(file)
            .map_err(|e| e.map_input(|input| std::str::from_utf8(input).unwrap()))
            .unwrap();

        assert_eq!(data.elements, elements);
    }
}
