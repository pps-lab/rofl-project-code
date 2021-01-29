
#[derive(Clone, Debug)]
pub enum EncModelParamType {
    Plain
}
#[derive(Clone)]
pub enum EncModelParams {
    Plain(PlainParams)
}

// Lubu: I started implementing the diffrent version of EncParams with enums instead of traits 
// due to object safty problems 
// https://www.mattkennedy.io/blog/rust_polymorphism/
// https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md
// https://doc.rust-lang.org/book/ch17-02-trait-objects.html
impl EncModelParams {
    pub fn unity(param_type : &EncModelParamType, size : usize) -> Self {
        match param_type {
            EncModelParamType::Plain => {
                return EncModelParams::Plain(PlainParams {
                    content : vec![0.0; size],
                });
            }
        }
    }

    pub fn accumulate_other(&mut self, other: &Self) -> bool {
        match self {
            EncModelParams::Plain(params) => {
                if let EncModelParams::Plain(other_plain) = other {
                    return params.accumulate_other(&other_plain);
                }
                false
            }
        }
    }

    pub fn verify(&self) -> bool {
        match self {
            EncModelParams::Plain(params) => {
                return true;
            }
        }
    }

    pub fn verifiable(&self) -> bool {
        match self {
            EncModelParams::Plain(params) => {
                return false;
            }
        }
    }

    pub fn extract(&self) -> Option<Vec<f32>> {
        match self {
            EncModelParams::Plain(params) => {
                return Some(params.content.clone());
            }
        }
    }

    pub fn length(&self) -> usize {
        match self {
            EncModelParams::Plain(params) => {
                return params.content.len()
            }
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        match self {
            EncModelParams::Plain(params) => {
                return bincode::serialize(&params.content).unwrap();
            }
        }
    }

    pub fn deserialize(param_type : &EncModelParamType, data : &Vec<u8>) -> Self {
        match param_type {
            EncModelParamType::Plain => {
                return EncModelParams::Plain(PlainParams {
                    content : bincode::deserialize(data).unwrap(),
                });
            }
        }
    }
}
#[derive(Clone)]
pub struct PlainParams {
    pub content : Vec<f32>,
}

impl PlainParams  {
    pub fn unity(size : usize) -> Self {
        return PlainParams {
            content : vec![0.0; size],
        };
    }
    
    pub fn accumulate_other(&mut self, other : &PlainParams) -> bool {
        if other.content.len() != self.content.len() {
            return false;
        }
        for i in 0..self.content.len() {
            self.content[i] += other.content[i]
        }
        return true;
    }

    pub fn serialize(&self) -> Vec<u8> {
        return bincode::serialize(&self.content).unwrap();
    }

    pub fn deserialize(data : &Vec<u8>) -> Self {
        return PlainParams {
            content : bincode::deserialize(data).unwrap(),
        };       
    }
}





#[cfg(test)]
mod tests {
    use super::*;

    fn test_helper(p1 : &mut EncModelParams, p2 : &EncModelParams, enc_type : &EncModelParamType) {
        p1.accumulate_other(&p2);
        p1.accumulate_other(&EncModelParams::unity(&enc_type, 10));
        assert_eq!(&p1.extract().unwrap()[..], &vec![2.0;10][..]);
        assert!(p1.verifiable() == false);
        let ser = p1.serialize();
        println!("{}" , ser.len());
        assert_eq!(&p1.extract().unwrap()[..], &EncModelParams::deserialize(&enc_type, &ser).extract().unwrap()[..]);
    }

    #[test]
    fn test_plain_params() {
        let mut p1 =  EncModelParams::Plain(PlainParams {
            content : vec![1.0; 10]
        });
        let p2 =  EncModelParams::Plain (PlainParams {
            content : vec![1.0; 10]
        });
        test_helper(&mut p1, &p2, &EncModelParamType::Plain);
    }
}